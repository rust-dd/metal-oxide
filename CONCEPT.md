# Compilation and execution

metal-oxide compiles Rust compute kernels to Metal Shading Language (MSL). Kernels are
written in ordinary crates; at build time the kernel macros hand their source to the
metal-oxide compiler, Apple's compiler builds the shader library, and the host launches the
kernels through generated functions and the metal-oxide runtime.

This document follows `vec_add` through the compiler, ABI, launch functions,
and execution. The example source is in [`examples/vec-add`](examples/vec-add).

## Build and runtime boundaries

```text
   [#[kernel] item in a host crate]
             |
   [kernel macro: device crate]
             |
        [rustc: MIR]
             |
       [importer: IR]
             |
         [codegen] -------------> ABI
             |                     |
            MSL                    |
             |                     |
      [Apple compiler]   [kernel macro: launch functions]
             |                     |
         .metallib                 |
             +---------------------+
             |
   [Rust host + runtime]
             |
 [Classic Metal / Metal 4]
```

There are three stages:

| Stage | Input | Output |
| --- | --- | --- |
| Kernel build | The item's source as a `no_std` device crate, or a kernel crate, with their device dependencies | MSL, ABI, and a `.metallib` |
| Host build | The host crate, whose kernel macros embed the library and generate launch functions | A native macOS executable |
| Execution | The executable, GPU buffers, and launch configuration | GPU commands and results |

`cargo build` runs the first two: the kernel macros run the kernel build inside the host
build. The compiler uses a pinned nightly rustc; the host and runtime build with stable
Rust. The runtime depends on the artifact format and Metal bindings, with no dependency on
`rustc_private`. The kernel build uses Cargo's release profile; `--release` selects the
host profile separately.

## Rust kernel and entrypoint marker

A kernel is an ordinary Rust function in the host crate:

```rust
use metal_oxide::device::{DisjointBuffer, ReadBuffer, thread_index};

/// Adds two buffers element by element; threads past the end of a buffer write nothing.
#[metal_oxide::kernel]
pub fn vec_add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    let a = a.as_slice();
    let b = b.as_slice();
    let i = index.get();
    if let Some(out) = out.get_mut(&index)
        && i < a.len()
        && i < b.len()
    {
        *out = a[i] + b[i];
    }
}
```

On the host, `#[metal_oxide::kernel]` keeps the function in a hidden module, where rustc and
rust-analyzer check it, and replaces it with a launch function of the same name. For the
kernel build it copies the source file into a device crate under `target/metal/sources/`,
with every character outside the item blanked so that each span keeps its line and column,
and adds the device prelude. There `#[kernel]` adds an entrypoint marker that the compiler
can find and preserves the function body. `#[metal_oxide::kernels]` does the same for an
inline module, whose other items stay on the host too.

A safe `fn` kernel launches safely: the device API makes every operation that can race or
read out of bounds `unsafe`, so a safe kernel is safe by Rust's own checking, or by its
author's promise for the `unsafe` blocks it contains. An `unsafe fn` kernel's launch function
is `unsafe`, and its callers uphold the contract in its doc comment.

`#[kernel(requires(a.len() == out.len(), b.len() == out.len()))]` lists relations between
buffer lengths, integer parameters and literals, with `+`, `-` and `*`. Every launch checks
them before it encodes and returns `Error::Contract` with the relation as written when one
fails. The compiler assumes those that compare plain lengths, parameters and literals at
the kernel's entry, so the unguarded `*out = a.as_slice()[i] + b.as_slice()[i]` after
`get_mut` needs no runtime check either.

`ReadBuffer`, `WriteBuffer`, and `ReadWriteBuffer` are device handles with explicit element access.
They avoid handing every GPU thread an exclusive Rust reference to the same
whole buffer. Read-write handles support in-place algorithms; their launch functions
borrow the buffer exclusively, while the caller still ensures race-free GPU access.
Inside the kernel entrypoint, a buffer parameter's `len()` is the element count the host
bound, and `ReadBuffer::as_slice()` views it as a checked `&[T]`. A `DisjointBuffer`
hands each thread `&mut` to only its own element: `get_mut(&thread_index())` returns it,
or `None` past the end, and the witness cannot be forged, copied or cloned. The index, which
`get` returns, is `block_idx().x * block_dim().x + thread_idx().x`, so a kernel that reads it
accepts only one-dimensional launches.

Primitive scalar buffers expose shared Metal storage directly through `read` and
`write`, which wait for pending GPU work first: awaiting them never blocks the thread,
and `wait` does. Owned records, arrays, and tuples retain explicit `GpuValue` codecs
and a CPU mirror. Scalars include `u64`
with eight-byte ABI alignment, wrapping arithmetic and saturating f32 casts.

Thread coordinates, math, and cooperative operations use bodyless
`extern "Rust"` declarations behind the device API. Rustc type-checks the
wrappers and calls. The importer checks the declaration's crate, module, and
signature through its resolved `DefId`, then emits the corresponding IR
operation. On the host, where kernels are only checked, these functions panic.

The device `math` module provides f32 `sqrt`, `fma`, `ln`, `exp`, `sin`,
`cos`, `sin_cos`, `powf`, `tanh`, `atan`, `floor`, `round`, `ceil`, and `trunc`.
Angles are radians; `ln` is natural log and `powf` takes a floating exponent.
`sin_cos` returns both values from one Metal `sincos` call. Transcendental results follow Metal's
precise accuracy bounds by default. Subnormal arithmetic may flush to zero;
arithmetic NaN payloads are unspecified. Default compilation keeps safe math
and disables implicit contraction. `fma` requests fusion explicitly.

Generic buffer and threadgroup methods adapt concrete Rust type/const arguments
to the memory declarations. The importer lowers these device adapters directly
to typed buffer operations, preserving their access and address space. Erased
pointers stay inside the adapter; they are not part of the kernel ABI or IR.
`F16::from_bits` and `F16::to_bits` remain ordinary const Rust bit conversions.

## Rust frontend and MIR

The compiler runs rustc through `rustc_driver`. Rustc expands macros, resolves
names, checks types and borrowing, and produces MIR: Mid-level Intermediate
Representation. MIR expresses operations on typed locals and control flow
between basic blocks.

The driver collects marked kernel entries and reachable concrete function
instances after analysis. This includes resolving type and const arguments for
supported generic helpers and device methods.

The kernel and its device dependencies use the custom
[`metal64-unknown-none` target](crates/metal-oxide-build/targets/metal64-unknown-none.json), with matching
`core` metadata. Proc macros build for the CPU host. Device layout queries use
the device target rather than assuming the host target is the GPU target.

For `vec_add`, rustc emits operations such as these. The excerpts omit storage
lifetime statements, declarations, and unrelated blocks:

```text
bb1: {
    _6 = &_1;
    _5 = metal_oxide_device::ReadBuffer::<f32>::as_slice(move _6) -> [return: bb2, unwind unreachable];
}

bb4: {
    _12 = &mut _3;
    _14 = &_4;
    _13 = &(*_14);
    _11 = metal_oxide_device::DisjointBuffer::<f32>::get_mut(move _12, move _13) -> [return: bb5, unwind unreachable];
}

bb5: {
    _15 = discriminant(_11);
    switchInt(move _15) -> [1: bb6, otherwise: bb16];
}

bb10: {
    _26 = copy _9;
    _27 = PtrMetadata(copy _5);
    _28 = Lt(copy _26, copy _27);
    assert(move _28, "index out of bounds: the length is {} but the index is {}", move _27, copy _26) -> [success: bb11, unwind unreachable];
}

bb11: {
    _25 = copy (*_5)[_26];
}
```

Here `_4` is the thread index, `_5` the slice of `a`, and `_9` is `i`. `get_mut`
returns an `Option<&mut f32>` whose discriminant `bb5` tests. Indexing keeps rustc's
bounds assertion against the slice length, `PtrMetadata(_5)`. Rustc has resolved the
method calls to concrete `ReadBuffer::<f32>::as_slice` and
`DisjointBuffer::<f32>::get_mut` instances.

MIR gives the importer an already checked Rust program with explicit operations.
Reading Rust syntax directly would require another frontend to determine types,
resolve calls and operator implementations, and lower control flow. For example,
`x + y` can be integer addition, floating-point addition, or a user-defined
`Add` implementation.

The compiler reads typed MIR through `tcx.instance_mir(instance.def)`. A textual
MIR dump is useful for inspection; it is not the compiler's input format.

## Import into the GPU IR

The importer constructs a `metal_oxide_ir::Module` from those rustc bodies.
This is a Rust data structure containing concrete functions, typed locals,
basic blocks, source positions, and explicit GPU operations.

For this kernel, the parameter types become:

| Parameter | IR type | Access | Address space |
| --- | --- | --- | --- |
| `a` | Buffer of `F32` | Read | Device |
| `b` | Buffer of `F32` | Read | Device |
| `out` | Buffer of `F32` | Write | Device |

The importer also replaces recognized calls:

| MIR operation | GPU IR operation |
| --- | --- |
| `thread_idx()` | `Coordinates(ThreadIdx)` |
| `block_idx()` | `Coordinates(BlockIdx)` |
| `block_dim()` | `Coordinates(BlockDim)` |
| `ThreadIndex::get()` | `ThreadIndex`; the witness itself is zero-sized |
| `DisjointBuffer::get_mut(index)` | the element's `Ref` if `ThreadIndex` is below the buffer's `Len` |
| `load_unchecked(buffer, index)` | `BufferLoad` |
| `store_unchecked(buffer, index, value)` | `BufferStore` |
| `buffer.len()`, `buffer.as_slice()` | `Len`, `Slice` of the parameter |
| `Iterator::next` on an integer `Range` or on `chunks_exact(n)` | expanded inline: a bound check, then the next value or chunk, so proofs keep loop bounds and chunk lengths |
| Other iterator methods and adapters (`iter`, `zip`, `map`, `fold`, `sum`, `rev`, `step_by`, ...) | core's own MIR, raw pointers included |
| Raw pointers, `NonNull<T>`, `NonZero<T>` | pointers and integers; `Offset`, `OffsetFrom`, `PointerCast` and `SliceFromParts` for pointer arithmetic |
| A tuple, record or array of references | an aggregate whose pointers keep the address spaces of the values that built it |
| An enum value, such as `Some(x)` | `Variant`: its tag and fields in payload slots that variants share by type |
| `match` on an enum | the tag, then a `Switch` |
| A closure and its calls | a record of its captures; `Fn*` calls reach its body with the argument tuple spread |
| `PhantomData`, fn items, closures without captures | unit: zero-sized values have no storage |
| Boolean `switchInt` | `Branch` |

With copy-only temporaries collapsed, the kernel has the following operations.
This is abbreviated notation for the IR data, not a device language:

```text
entry:
    i:       Usize      = ThreadIndex
    a:       &[F32]     = Slice(a)
    b:       &[F32]     = Slice(b)
    out:     &mut [F32] = Slice(out)
    has_out: Bool       = Binary(Lt, i, Len(out))
    Branch(has_out, element, done)

element:
    slot:    &mut F32   = Ref((*out)[i])
    has_a:   Bool       = Binary(Lt, i, Len(a))
    Branch(has_a, second, done)

second:
    has_b:   Bool       = Binary(Lt, i, Len(b))
    Branch(has_b, write, done)

write:
    Assert(Lt(i, Len(a)), Bounds)
    x: F32 = (*a)[i]
    Assert(Lt(i, Len(b)), Bounds)
    y: F32 = (*b)[i]
    *slot = Binary(Add, x, y)
    Goto(done)

done:
    Return
```

The IR validator checks types, initialization, call graphs, address spaces, and
cooperative participation. Codegen structures supported control flow into MSL.
Unsupported operations produce diagnostics. MIR assertions remain in the IR
until a typed constant or range proof establishes that they cannot fail.
The guards prove both bounds assertions, so `vec_add` has no runtime checks.
Launch validation keeps `grid_dim * block_dim` within `u32` on every axis, so
`block_idx * block_dim + thread_idx` needs no overflow check; under a required block,
neither does `block_idx * B + thread_idx` for a constant `B` up to the block size.
`ThreadIndex` is Metal's `thread_position_in_grid.x`; the runtime launches a kernel
that reads it only in one dimension, where no two threads share it.
Metal has no device trap, so each unproved assertion, each literal `panic!` or
`assert!` and each `unwrap` or `expect` becomes a runtime check: a failing thread stores
the check in its dispatch's failure word and returns. Checks report Rust's location, the
caller of a `#[track_caller]` function included, and Rust's message; a `Result`'s error
is not formatted. A check before a barrier or SIMD collective
must be proved or uniform, so that no thread waits for one that has returned. `log!`
lowers to Metal shader logging with typed `printf` specifiers.

The compiler writes a diagnostic text dump of this structure to
`kernels.oxide-ir`.

## MSL and ABI generation

Codegen produces two outputs from the imported program:

| Function | Input | Output |
| --- | --- | --- |
| `metal_oxide_codegen::Codegen::new` | IR module | Validated codegen context and kernel parameter bindings |
| `Codegen::emit` | Codegen context | MSL source |
| `Codegen::abi` | The same context | In-memory `Abi` |

The compiler restores parameter names from MIR debug information before writing
the ABI.

The codegen context assigns binding indices once. MSL emission and ABI generation
use that shared kernel interface. The MSL emitter also reads the IR for types and
function bodies; it does not read `abi.json`.

### MSL

MSL is Apple's shader language. The generated source declares buffer address
spaces, binding slots, and builtin coordinates.

The emitter retains extra locals and a builtin context, and precedes statements with
`#line` directives, so Apple's compiler diagnostics and GPU debuggers name the Rust
source line; generated declarations keep their `kernels.metal` lines. With the copy-only
locals and directives removed and identifiers shortened, `vec_add` is equivalent to:

```cpp
#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

kernel void vec_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint3 thread_position [[thread_position_in_grid]],
    const device uint* gate [[buffer(3)]],
    constant ulong* lengths [[buffer(4)]]
) {
    ulong i = thread_position.x;
    if (i < lengths[2] && i < lengths[0] && i < lengths[1]) {
        if (*gate != 0u) { return; }
        out[i] = a[i] + b[i];
    }
}
```

The host supplies three arguments: `a`, `b`, and `out`. The runtime binds the gate
word and the table of their element counts after them, and Metal supplies the
coordinate parameters from the dispatch:

| Device function | MSL builtin |
| --- | --- |
| `thread_idx()` | `thread_position_in_threadgroup` |
| `block_idx()` | `threadgroup_position_in_grid` |
| `block_dim()` | `threads_per_threadgroup` |
| `grid_dim()` | `threadgroups_per_grid` |
| `thread_index()` | `thread_position_in_grid` (x) |

### ABI

The ABI is the host-kernel argument contract. Its rules and supported
representations are defined by the project; each kernel's description is
generated during the kernel build.

For `vec_add`, `abi.json` contains:

```json
{
  "version": 4,
  "math_mode": "precise",
  "required_features": [],
  "types": [
    { "size": 4, "alignment": 4, "kind": { "kind": "scalar", "scalar": "f32" } }
  ],
  "signatures": [
    [
      { "name": "a", "binding": 0, "ty": { "kind": "buffer", "element": 0, "stride": 4, "access": "read" } },
      { "name": "b", "binding": 1, "ty": { "kind": "buffer", "element": 0, "stride": 4, "access": "read" } },
      { "name": "out", "binding": 2, "ty": { "kind": "buffer", "element": 0, "stride": 4, "access": "write" } }
    ]
  ],
  "kernels": [
    {
      "name": "vec_add",
      "signature": 0,
      "required_block": null,
      "gate_binding": 3,
      "fallible": false,
      "lengths": true,
      "one_dimensional": true,
      "safe": true,
      "requires": []
    }
  ]
}
```

Kernels share two tables. `types` holds each distinct layout once, and nested
layouts refer to earlier entries. `signatures` holds each distinct parameter
list. The in-memory `Abi` keeps every kernel's parameters and layouts inline.
Each kernel also records its gate slot, whether it can fail a check (it then takes
the failure word next), whether it reads buffer lengths (it then takes their table
next), whether it reads its thread index, whether it is a safe `fn`, and its `requires`
relations, each as written and as a term over parameter indices.

Buffer handles become Metal bindings. Values and buffer elements carry a
canonical layout with size, alignment, field offsets, array stride, and padding.
Records that the host shares with kernels implement `GpuValue` to encode/decode those bytes.
Scalars also use this representation; a `u32` value is four bytes. The host does not copy
the memory layout of the Rust device handles into the shader.

Build-time and runtime consumers use this description:

| When | Consumer | Use |
| --- | --- | --- |
| Artifact build | `metal-oxide-build` | Embed the ABI in `manifest.json` |
| Host build | Kernel macros | Check the launch functions against it; define a kernel crate's host records |
| Artifact loading | Runtime | Validate the manifest and the launch functions' ABI digest; retain each kernel's ABI |
| Kernel launch | Runtime | Check argument count, scalar/element types, access modes, any required block shape, the one-dimensional launch of a thread-index kernel, and the kernel's `requires` relations |

The Apple compiler compiles the MSL declarations, including their binding
attributes. It does not generate our ABI file or the launch functions.

## Launch functions

The kernel macros generate one function per kernel, with the kernel's name and visibility,
that takes the `Device`, the launch configuration and the kernel's arguments and returns
its `Launch` operation. They derive it from the signature and check it against the built
ABI: the kernels, their parameters' names, kinds and access, safety and block shape. The
crate embeds the manifest and library in a hidden static. A `Device` loads an embedded
library on its first launch, verifies the ABI digest the functions were generated for, and
keeps the library and each kernel's pipeline for all of its clones. A load or pipeline
failure is returned when that operation runs.

The ABI determines the signature: read buffers accept a shared borrow, an exclusive
borrow, an owned buffer or an `Arc` (`ReadArgument`); write, read-write and atomic
buffers accept an exclusive borrow or an owned buffer (`WriteArgument`). Scalar
parameters use their canonical Rust types. Parameter order determines Metal binding
slots. The operation returns its arguments in the form they were passed, so owned
buffers come back after the GPU work. `sequence(|batch| ...)` pushes several
operations into one ordered command buffer. Required block shapes remain encoded in
the configuration type and checked by the runtime.

A `#[kernels]` module also gets `module(&device)`, its library as a `Module` for pipelines
that code encodes itself. Its `#[repr(C)]` records that kernels take implement `GpuValue`
with their Rust layout, which launches check against the ABI. With `path = "../kernels"`
the module launches the kernels of a separate `no_std` kernel crate and defines a host
struct for each record they take, whose layout const assertions check against the ABI.

The macros track their inputs on stable Rust: the expansion includes the library, the
compiler and a kernel crate's files through `include_bytes!` and reads the variables that
steer kernel builds through `option_env!`, so Cargo rebuilds the crate when one changes.
The artifact files a kernel build writes are dated 1970: Cargo would take files that appear
during a build for changes and build the crate a second time.
Under rust-analyzer the macros generate the launch functions without a kernel build.

### Saved artifacts

After each kernel build the macros save the artifact beside the crate's sources, in
`metal/<file>/<item>-<math mode>/`: `manifest.json`, `kernels.metallib`, and `inputs.json`,
which records what it was built from: a digest of the item's source with its file, name and
dependencies, or of the kernel crate's path; the math mode; the registry and git packages
from `Cargo.lock`; and digests of the path packages' files. The saved artifact stands in for
a kernel build

- in a package that Cargo fetched from a registry, a git repository or a vendor directory,
  whose sources stay as published;
- without the kernel toolchain, while its recorded inputs are unchanged;
- with `METAL_OXIDE_ARTIFACTS=locked`, which builds no kernels and makes a missing or stale
  saved artifact an error, also on hosts without Metal, which otherwise skip kernels.

A build with the toolchain rewrites a saved artifact whose inputs changed. Commit and
publish the `metal/` directory. A removed kernel leaves its directory behind; deleting
`metal/` and building again keeps only the current ones.

## Apple compilation and artifact

`metal-oxide-build`, the library behind the kernel macros, invokes the Apple compiler in
two steps:

```sh
xcrun -sdk macosx metal \
    -std=metal3.2 -mmacosx-version-min=15.0 \
    -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise \
    -ffp-contract=off -O2 \
    -c kernels.metal -o kernels.ir
xcrun -sdk macosx metal kernels.ir -o kernels.metallib
```

When `xcrun` cannot locate the compiler, it also resolves the installed
Metal Toolchain component. The default precise math mode uses safe math,
precise floating-point functions, and disabled floating-point contraction.

`#[kernels(math = "fast")]` selects Metal's relaxed
arithmetic mode, fast f32 functions and fast contraction. It permits reciprocal
approximations and reassociation, including fusion across statements. Arithmetic
honors NaN and infinity; zero signs are unspecified. Float-to-integer casts still
saturate and map NaN to zero. Integer checks and synchronization proofs are
independent of the math mode.

Fast function accuracy follows MSL Table 8.2: sin/cos have absolute error <= 2^-13
within [-pi, pi], with weaker accuracy outside that interval. Ln requires positive
inputs (absolute error <= 2^-21 within [0.5, 2], <= 3 ULP elsewhere); exp allows
3 + floor(abs(2*x)) ULP. Powf uses an exp2/log2 approximation, with no precise-mode
promise for negative bases or 0^0. Tanh uses an exponential approximation; atan
allows 5 ULP. Precise-mode edge-case guarantees do not extend to fast functions.
The [Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
defines the full function contracts.

Generated MSL carries the numerical pragmas and explicit `precise::`/`fast::`
calls, so `Module::from_source` uses the same contract as offline compilation.
The mode is required in the ABI and participates in the Rust build cache,
artifact fingerprint and saved artifact name. Mismatched Metal flags,
unknown modes and artifacts missing this field are rejected. Rebuild development
artifacts after format changes, including the shared ABI tables; the pre-release
ABI version remains 4.

The artifact is stored under `target/metal/<build-hash>/`:

| File | Created by | Use |
| --- | --- | --- |
| `kernels.oxide-ir` | MIR importer/output stage | Diagnostic dump of our IR |
| `rustc-args.json` | Compiler driver | Effective kernel compiler arguments, included in the build identity |
| `kernels.metal` | MSL codegen | Input to the Apple compiler; inspectable shader source |
| `abi.json` | ABI codegen | Standalone kernel contract |
| `kernels.ir` | Apple compiler | Apple's intermediate code used to build the library |
| `kernels.metallib` | Apple compiler | Loaded by the runtime |
| `manifest.json` | `metal-oxide-build` | ABI, requirements, file hashes, and build identity |

`kernels.oxide-ir` and `kernels.ir` are different representations produced
by different compilers. The host embeds `manifest.json` and
`kernels.metallib`; diagnostic IR is for inspection. `cargo metal inspect --emit ir`,
`--emit msl` and `--emit air` print the first two and the AIR as text.

The cache identity includes sources and dependencies, compiler and toolchain
versions, target configuration, enabled features, and compilation options.
Cache reuse checks the generated files against the manifest.

## Host build and launch

A minimal macOS host is the kernel above and the code that launches it:

```rust
use metal_oxide::{Device, LaunchConfig, Operation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 257_u32;
    let device = Device::system_default()?;

    let a = device.buffer_from_slice(&vec![1.0_f32; n as usize])?;
    let b = device.buffer_from_slice(&vec![2.0_f32; n as usize])?;
    let mut out = device.buffer_zeroed::<f32>(n as usize)?;
    let config = LaunchConfig::<256>::for_elements(n)?;

    vec_add(&device, config, &a, &b, &mut out).wait()?;

    assert!(out.read().wait().iter().all(|&value| value == 3.0));
    Ok(())
}
```

`LaunchConfig::<256>` sets 256 threads per block. For 257 elements,
`for_elements` produces two blocks, so 512 GPU threads execute. Threads 257 through
511 find no element of `out`, so `get_mut` returns `None` and they write nothing.

The launch configuration controls dispatch geometry; the bound buffers' lengths
bound the kernel's accesses. The launch function takes the device, the configuration
and the three kernel arguments.

When an entrypoint declares `#[kernel(block = (256, 1, 1))]`, its ABI records
that shape and the launch function accepts the matching `LaunchConfig` type.
`vec_add` has no required shape, so it accepts const or dynamic
configuration. Host const generics alone do not specialize the shader.

The runtime performs these steps:

1. `Device::system_default` requires macOS 15, creates the GPU context and
   selects a native backend.
2. `vec_add` returns a `Launch` operation that holds the device, the embedded library,
   the configuration and the arguments; nothing is loaded or encoded yet.
3. On the device's first launch from that library, it validates the manifest's format
   and requirements, verifies the library bytes and the ABI digest, and loads the bytes
   into Metal. It then creates the native `vec_add` pipeline and retains that
   entrypoint's ABI. Later launches through any clone of the device reuse both.
5. `wait` (or the first poll of its future) creates a `Batch`, which binds the three
   ordered arguments, checks resource context, ABI types/access, block
   requirements, the one-dimensional launch of a kernel that reads its thread index,
   and dispatch limits, and records the dispatch.
6. The batch commits. It places each recorded dispatch in the first phase after every
   dispatch that wrote a buffer it uses or read a buffer it writes, and encodes the
   phases with a barrier between them; the dispatches of a phase may run concurrently.
   A dispatch binds a gate word, for a kernel that can fail a check its own failure
   word, and for a kernel that reads buffer lengths a table of its arguments' element
   counts. A kernel tests its gate once, at the latest point that precedes all of its
   memory accesses, synchronization and logs, and returns once a dispatch of an
   earlier phase has failed. The dispatches of a phase share a gate that none of them
   writes: zero, an earlier failure word, or a snapshot of several, which one thread
   per word sets after the previous barrier.
7. `wait` blocks until GPU completion, and the future resolves then; either reports
   command errors, or `Error::KernelFailed` with the kernel, dispatch, Rust location
   and message of the failed check. A batch that reads buffers written by operations
   still running when it commits gates its generated kernels on their words as well,
   and reports their failure as `Error::DependencyFailed`.
8. The host reads the output buffer.

The runtime's `Module` holds a loaded Metal library and ABI metadata.
`metal_oxide_ir::Module` holds the compiler's program representation.

The runtime checks the call interface and the `requires` relations against the ABI. For
an `unsafe fn` kernel, its documented bounds, aliasing, race, and participation
requirements remain the caller's responsibility; a kernel that indexes with
`load_unchecked` must keep its indices in bounds.

## Debugging

`cargo metal run|test --validate` runs the host under Metal API and shader validation.
`cargo metal run|test --capture FILE.gputrace` records the host's GPU work for Xcode,
which replays the dispatches and attributes their instructions to Rust lines:

1. The kernels build into a separate artifact under `target/metal/` whose library holds
   line tables and its compile inputs (`-gline-tables-only -frecord-sources`), with
   `-fdebug-compilation-dir` at the kernel workspace so the relative paths of the
   `#line` directives resolve. Saved artifacts never carry them.
2. The host runs with `METAL_CAPTURE_ENABLED=1` and `METAL_OXIDE_CAPTURE` set to the
   absolute trace path. The first `Device` of the process captures the native device,
   which covers every queue of both backends, and stops when its last clone, including
   those its pipelines hold, drops.

Metal does not replace an existing trace, and later devices of the process are not
recorded, so give `test` a filter that selects the test to capture. Kernels from saved
artifacts, as in a dependency, are captured without Rust lines.

## Native resource ownership

`Device::all` creates independently validated contexts in native enumeration order;
select one with `into_iter().nth(index)`. The order is not a persistent identity.
`recommended_max_working_set_size` returns an advisory byte budget, not currently
free memory. Buffer allocation failure is `Error::AllocationFailed`; terminal
command memory exhaustion is `Error::OutOfMemory` on both backends.

`Device::from_native` retains an `objc2-metal` device and creates its command queues.
`as_native` borrows that device for external resource creation. Unsafe
`Buffer::from_native` consumes retained, initialized, shared, hazard-tracked storage
on the same physical device. It validates the canonical element layout, extent
and alignment. External GPU work must already be complete, storage must remain
non-purgeable and non-aliasable, and external aliases must stay idle until runtime
CPU views and encoded/pending GPU accesses end. The runtime may retain, release and
read immutable properties of the native buffer on any thread. Runtime contexts
remain distinct.

Consuming `Buffer::into_native` blocks until pending accesses finish and encodes
dirty custom values before returning the native buffer. Primitive buffers preserve
their storage address; records use their existing canonical codec. Dropping or
forgetting a running operation does not bypass export synchronization. Merely
dropping the buffer does not complete GPU work. Command errors are reported only by
the operation that ran them.

`Device`, `Module`, `Pipeline`, `Buffer`, `Argument`, `Launch` and the operation
futures are `Send + Sync` when their arguments are. `Device` and `Pipeline` are cheap
handles: device clones share its four command queues. An operation commits when it is first
polled or waited and runs after the operations committed before it that use its buffers: it
joins the queue of that work, and waits for work on other queues through events. Without such
work it takes the least busy queue. Completion arrives in commit order per queue, so
independent work is not held behind long work on another queue. A `Batch` records its
dispatches and encodes them when its operation commits. CPU views on
any thread wait for every registered GPU access; GPU writes require an exclusive
borrow or ownership, so codec buffers decode a GPU write once even when several
threads read concurrently. [The threading tests](crates/metal-oxide/tests/threads.rs)
cover shared devices and pipelines, cross-thread readback, awaiting and dropping.

## Classic Metal and Metal 4

The Rust-to-MSL build and artifact format are shared. Native API selection
happens when creating the `Device`. By default, macOS 26 and a GPU supporting
the Metal 4 family select Metal 4; otherwise the runtime selects classic Metal.

The selected variant controls pipeline creation and batch encoding:

| Operation | Classic Metal | Metal 4 |
| --- | --- | --- |
| Pipeline creation | Device compute-pipeline API | `MTL4Compiler` |
| Command encoding | `MTLComputeCommandEncoder` | `MTL4ComputeCommandEncoder` and command allocator |
| Arguments | `setBuffer`; values up to 4 KiB copied with `setBytes`, larger values retained in buffers | GPU addresses in an immutable `MTL4ArgumentTable`; values packed into aligned, immutable storage per dispatch |
| Resource residency | Classic resource management | Explicit residency set |
| Dispatch ordering | A serial encoder for a chain; otherwise a concurrent encoder with buffer barriers between phases | Encoder barriers between phases; a queue barrier when the batch follows work on its queue |
| Work on other queues | Each command buffer sets a pooled event when it finishes; dependents wait for it | The queue signals a timeline event after its committed work on demand; dependents wait for it |
| Commit | Command-buffer commit and completion callback | Command-queue commit and feedback callback |

Both paths retain native resources until GPU completion and expose the same
`Device`, `Pipeline`, `Launch`, `Batch` and `Operation` API.
`METAL_OXIDE_BACKEND=classic` or `metal4` forces a path for verification;
forcing Metal 4 on an unsupported device returns an error.

## Implementation map

| Responsibility | Source |
| --- | --- |
| Kernel macros and launch functions | [`kernels/src/lib.rs`](crates/metal-oxide-kernels/src/lib.rs), [`launch.rs`](crates/metal-oxide-kernels/src/launch.rs), [`device.rs`](crates/metal-oxide-kernels/src/device.rs) |
| Entrypoint marker | [`metal-oxide-macros/src/lib.rs`](crates/metal-oxide-macros/src/lib.rs) |
| Rustc callbacks | [`compiler/src/driver.rs`](crates/metal-oxide-compiler/src/driver.rs) |
| Entries and concrete instances | [`compiler/src/collect.rs`](crates/metal-oxide-compiler/src/collect.rs), [`monomorphize.rs`](crates/metal-oxide-compiler/src/monomorphize.rs) |
| MIR import and intrinsic lowering | [`compiler/src/import/mod.rs`](crates/metal-oxide-compiler/src/import/mod.rs) |
| IR data and validation | [`ir/src/model.rs`](crates/metal-oxide-ir/src/model.rs), [`validate.rs`](crates/metal-oxide-ir/src/validate.rs) |
| MSL and ABI | [`codegen/src/emit.rs`](crates/metal-oxide-codegen/src/emit.rs), [`abi.rs`](crates/metal-oxide-codegen/src/abi.rs) |
| Compiler output files | [`compiler/src/output.rs`](crates/metal-oxide-compiler/src/output.rs) |
| Artifact model and validation | [`artifact/src/model.rs`](crates/metal-oxide-artifact/src/model.rs), [`validate.rs`](crates/metal-oxide-artifact/src/validate.rs) |
| Kernel build and Apple compiler | [`build/src/rust.rs`](crates/metal-oxide-build/src/rust.rs), [`toolchain.rs`](crates/metal-oxide-build/src/toolchain.rs) |
| Manifest and cache | [`build/src/compile.rs`](crates/metal-oxide-build/src/compile.rs), [`cache.rs`](crates/metal-oxide-build/src/cache.rs) |
| Device crates, reuse and saved artifacts | [`build/src/source.rs`](crates/metal-oxide-build/src/source.rs), [`kernel_crate.rs`](crates/metal-oxide-build/src/kernel_crate.rs), [`saved.rs`](crates/metal-oxide-build/src/saved.rs), [`recorded.rs`](crates/metal-oxide-build/src/recorded.rs) |
| CLI | [`cargo-metal/src/setup.rs`](crates/cargo-metal/src/setup.rs), [`run.rs`](crates/cargo-metal/src/run.rs), [`inspect.rs`](crates/cargo-metal/src/inspect.rs) |
| Library loading and launch validation | [`runtime/metal/library.rs`](crates/metal-oxide/src/metal/library.rs), [`module.rs`](crates/metal-oxide/src/metal/module.rs), [`pipeline.rs`](crates/metal-oxide/src/metal/pipeline.rs) |
| Operations and native paths | [`runtime/metal/operation.rs`](crates/metal-oxide/src/metal/operation.rs), [`batch.rs`](crates/metal-oxide/src/metal/batch.rs), [`backend.rs`](crates/metal-oxide/src/metal/backend.rs) |
