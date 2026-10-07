# Compilation and execution

metal-oxide compiles Rust compute kernels to Metal Shading Language (MSL).
Apple's compiler builds the shader library, and a Rust host loads and launches
its kernels through the metal-oxide runtime.

This document follows `vec_add` through the compiler, ABI, generated bindings,
and execution. The example source is in [`examples/vec-add`](examples/vec-add).

## Build and runtime boundaries

```text
       [Rust kernel]
             |
        [rustc: MIR]
             |
       [importer: IR]
             |
         [codegen] ----> ABI + bindings
             |                 |
            MSL                |
             |                 |
      [Apple compiler]         |
             |                 |
         .metallib             |
             +-----------------+
             |
   [Rust host + runtime]
             |
 [Classic Metal / Metal 4]
```

There are three stages:

| Stage | Input | Output |
| --- | --- | --- |
| Kernel build | A separate `no_std` Rust crate and its device dependencies | MSL, ABI, host bindings, and a `.metallib` |
| Host build | Ordinary Rust application code and the generated bindings | A native macOS executable |
| Execution | The executable, artifact, GPU buffers, and launch configuration | GPU commands and results |

`cargo metal run -p vec-add` coordinates all three. The compiler uses a pinned
nightly rustc; the host and runtime build with stable Rust. The runtime depends on
the artifact format and Metal bindings, with no dependency on `rustc_private`.
The kernel build uses Cargo's release profile; `--release` selects the host
profile separately.

The host package identifies its kernel crate in `Cargo.toml`:

```toml
[package.metadata.metal]
kernels = "../kernels/Cargo.toml"
```

## Rust kernel and entrypoint marker

The kernel crate contains ordinary Rust:

```rust
#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

/// # Safety
///
/// Buffers contain at least n elements. Output does not overlap either input.
/// The launch is one-dimensional, with one writer per output index.
#[kernel]
pub unsafe fn vec_add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, out: WriteBuffer<f32>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;

    if i < n {
        // SAFETY: i < n; the caller supplies valid buffers and independent output indices.
        unsafe {
            out.store_unchecked(i, a.load_unchecked(i) + b.load_unchecked(i));
        }
    }
}
```

`#[kernel]` adds an entrypoint marker that the compiler can find. The macro
preserves the function body. MSL and host binding generation happen later.

`ReadBuffer` and `WriteBuffer` are device handles with explicit element access.
They avoid handing every GPU thread an exclusive Rust reference to the same
whole buffer.

Thread coordinates and buffer operations are device intrinsics. Their Rust
definitions let rustc type-check calls; CPU execution of the intrinsic bodies
panics. The MIR importer recognizes the definitions and lowers their calls to
GPU operations.

## Rust frontend and MIR

The compiler runs rustc through `rustc_driver`. Rustc expands macros, resolves
names, checks types and borrowing, and produces MIR: Mid-level Intermediate
Representation. MIR expresses operations on typed locals and control flow
between basic blocks.

The driver collects marked kernel entries and reachable concrete function
instances after analysis. This includes resolving type and const arguments for
supported generic helpers and device methods.

The kernel and its device dependencies use the custom
[`metal64-unknown-none` target](targets/metal64-unknown-none.json), with matching
`core` metadata. Proc macros build for the CPU host. Device layout queries use
the device target rather than assuming the host target is the GPU target.

For `vec_add`, rustc emits operations such as these. The excerpts omit storage
lifetime statements, declarations, and unrelated blocks:

```text
bb0: {
    _8 = metal_oxide_device::block_idx() -> [return: bb1, unwind unreachable];
}

bb2: {
    _9 = copy (_10.0: u32);
    _6 = Mul(move _7, move _9);
    _12 = metal_oxide_device::thread_idx() -> [return: bb3, unwind unreachable];
}

bb3: {
    _11 = copy (_12.0: u32);
    _5 = Add(move _6, move _11);
    _14 = copy _5;
    _15 = copy _4;
    _13 = Lt(move _14, move _15);
    switchInt(move _13) -> [0: bb8, otherwise: bb4];
}
```

Here `_4` is `n`, `_5` is `i`, and the `.0` projections read the
`x` coordinate. Rustc has also resolved the buffer calls to concrete
`ReadBuffer::<f32>::load_unchecked` and `WriteBuffer::<f32>::store_unchecked`
methods.

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
| `n` | Scalar `U32` | Value | — |

The importer also replaces recognized calls:

| MIR operation | GPU IR operation |
| --- | --- |
| `thread_idx()` | `Coordinates(ThreadIdx)` |
| `block_idx()` | `Coordinates(BlockIdx)` |
| `block_dim()` | `Coordinates(BlockDim)` |
| `load_unchecked(buffer, index)` | `BufferLoad` |
| `store_unchecked(buffer, index, value)` | `BufferStore` |
| Boolean `switchInt` | `Branch` |

With copy-only temporaries collapsed, the kernel has the following operations.
This is abbreviated notation for the IR data, not a device language:

```text
coordinates:
    block:  Dim3 = Coordinates(BlockIdx)
    shape:  Dim3 = Coordinates(BlockDim)
    thread: Dim3 = Coordinates(ThreadIdx)
    offset: U32  = Binary(Mul, block.x, shape.x)
    i:      U32  = Binary(Add, offset, thread.x)
    active: Bool = Binary(Lt, i, n)
    Branch(active, write, done)

write:
    x: F32 = BufferLoad(a, i)
    y: F32 = BufferLoad(b, i)
    z: F32 = Binary(Add, x, y)
    BufferStore(out, i, z)
    Goto(done)

done:
    Return
```

The IR validator checks types, initialization, call graphs, address spaces, and
cooperative participation. Codegen structures supported control flow into MSL.
Unsupported operations produce diagnostics. An enabled MIR assertion remains in
the IR and is rejected by current codegen rather than silently discarded.

The compiler writes a diagnostic text dump of this structure to
`kernels.oxide-ir`.

## MSL and ABI generation

Codegen produces three outputs from the imported program:

| Function | Input | Output |
| --- | --- | --- |
| `metal_oxide_codegen::Codegen::new` | IR module | Validated codegen context and kernel parameter bindings |
| `Codegen::emit` | Codegen context | MSL source |
| `Codegen::abi` | The same context | In-memory `Abi` |
| `metal_oxide_codegen::bindings` | `Abi` | Rust host source |

The compiler restores parameter names from MIR debug information before writing
the ABI and bindings.

The codegen context assigns binding indices once. MSL emission and ABI generation
use that shared kernel interface. The MSL emitter also reads the IR for types and
function bodies; it does not read `abi.json`. The binding generator consumes the
in-memory ABI.

### MSL

MSL is Apple's shader language. The generated source declares buffer address
spaces, binding slots, and builtin coordinates.

The emitter retains extra locals and a builtin context. With those copy-only
locals collapsed and identifiers shortened, `vec_add` is equivalent to:

```cpp
#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

kernel void vec_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint3 thread_idx [[thread_position_in_threadgroup]],
    uint3 block_idx [[threadgroup_position_in_grid]],
    uint3 block_dim [[threads_per_threadgroup]],
    uint3 grid_dim [[threadgroups_per_grid]]
) {
    uint i = block_idx.x * block_dim.x + thread_idx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
```

The host supplies four arguments: `a`, `b`, `out`, and `n`. Metal supplies
the coordinate parameters from the dispatch:

| Device function | MSL builtin |
| --- | --- |
| `thread_idx()` | `thread_position_in_threadgroup` |
| `block_idx()` | `threadgroup_position_in_grid` |
| `block_dim()` | `threads_per_threadgroup` |
| `grid_dim()` | `threadgroups_per_grid` |

### ABI

The ABI is the host-kernel argument contract. Its rules and supported
representations are defined by the project; each kernel's description is
generated during the kernel build.

For `vec_add`, `abi.json` contains:

```json
{
  "version": 2,
  "required_features": [],
  "kernels": [
    {
      "name": "vec_add",
      "parameters": [
        {
          "name": "a", "binding": 0,
          "ty": { "kind": "buffer", "element": "f32", "access": "read" }
        },
        {
          "name": "b", "binding": 1,
          "ty": { "kind": "buffer", "element": "f32", "access": "read" }
        },
        {
          "name": "out", "binding": 2,
          "ty": { "kind": "buffer", "element": "f32", "access": "write" }
        },
        {
          "name": "n", "binding": 3,
          "ty": { "kind": "scalar", "scalar": "u32" }
        }
      ],
      "required_block": null
    }
  ]
}
```

Buffer handles become Metal bindings. Scalars use the ABI's defined scalar
representation; `n` is a four-byte `u32`. The host does not copy the memory
layout of the Rust device handles into the shader.

Build-time and runtime consumers use this description:

| When | Consumer | Use |
| --- | --- | --- |
| Kernel build | Binding generator | Produce typed host parameters and ordered `Argument` values |
| Artifact build | `cargo-metal` | Embed the ABI in `manifest.json` |
| Artifact loading | Runtime | Validate the manifest and retain each kernel's ABI |
| Kernel launch | Runtime | Check argument count, scalar/element types, access modes, and any required block shape |

The Apple compiler compiles the MSL declarations, including their binding
attributes. It does not generate our ABI file or Rust wrappers.

## Generated Rust bindings

`bindings.rs` is host Rust code. It contains a `Kernels` struct, a `load`
function, and synchronous and enqueue methods for each entrypoint.

For `vec_add`, the synchronous part is equivalent to the following, with
generated identifiers and formatting shortened:

```rust
use metal_oxide::{Argument, Buffer, Device, DynamicLaunchConfig, Module, Pipeline, Result};

pub struct Kernels<'a> {
    device: &'a Device,
    pipeline_0: Pipeline,
}

pub fn load(device: &Device, directory: impl AsRef<std::path::Path>) -> Result<Kernels<'_>> {
    let module = Module::from_artifact(device, directory)?;
    Ok(Kernels {
        device,
        pipeline_0: Pipeline::new(device, &module, "vec_add")?,
    })
}

impl Kernels<'_> {
    /// # Safety
    ///
    /// Buffers and launch geometry must satisfy vec_add's bounds and race contract.
    pub unsafe fn vec_add(
        &self,
        config: impl Into<DynamicLaunchConfig>,
        a: &Buffer<f32>,
        b: &Buffer<f32>,
        out: &mut Buffer<f32>,
        n: u32,
    ) -> Result<()> {
        // SAFETY: the caller supplies the kernel-specific bounds and race contract.
        unsafe {
            self.device.launch(
                &self.pipeline_0,
                config,
                &[
                    Argument::read(a),
                    Argument::read(b),
                    Argument::write(out),
                    Argument::u32(n),
                ],
            )
        }
    }
}
```

The ABI determines the signature: read buffers take shared borrows, the write
buffer takes an exclusive borrow, and `n` is `u32`. It also determines the
argument order: slice position zero is binding zero, and so on.

The generated `enqueue_vec_add` method takes a `Batch` and calls
`Batch::launch` with the same arguments. It lets the host encode the kernel
alongside other kernels in one ordered submission.

These wrappers are compiled into the host executable. Runtime loading reads the
manifest's ABI, while argument construction executes the already compiled
wrapper code.

## Apple compilation and artifact

`cargo-metal` invokes the Apple compiler in two steps:

```sh
xcrun -sdk macosx metal \
    -std=metal3.1 -mmacosx-version-min=15.0 \
    -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise \
    -ffp-contract=off -O2 \
    -c kernels.metal -o kernels.ir
xcrun -sdk macosx metal kernels.ir -o kernels.metallib
```

When `xcrun` cannot locate the compiler, the CLI also resolves the installed
Metal Toolchain component. The compilation uses safe math, precise floating-point
functions, and disabled floating-point contraction.

The artifact is stored under `target/metal/<build-hash>/`:

| File | Created by | Use |
| --- | --- | --- |
| `kernels.oxide-ir` | MIR importer/output stage | Diagnostic dump of our IR |
| `kernels.metal` | MSL codegen | Input to the Apple compiler; inspectable shader source |
| `abi.json` | ABI codegen | Standalone kernel contract |
| `bindings.rs` | Binding codegen | Compiled into the host |
| `kernels.ir` | Apple compiler | Apple's intermediate code used to build the library |
| `kernels.metallib` | Apple compiler | Loaded by the runtime |
| `manifest.json` | `cargo-metal` | ABI, requirements, file hashes, and build identity |

`kernels.oxide-ir` and `kernels.ir` are different representations produced
by different compilers. The runtime consumes `manifest.json` and
`kernels.metallib`; diagnostic IR is for inspection.

The cache identity includes sources and dependencies, compiler and toolchain
versions, target configuration, enabled features, and compilation options.
Cache reuse checks the generated files against the manifest.

## Host build and launch

`cargo-metal` passes `METAL_OXIDE_BINDINGS` and `METAL_OXIDE_ARTIFACT_DIR`
to the ordinary host Cargo build. The host includes the generated source and
uses the artifact path when loading kernels.

A minimal macOS host using the generated `vec_add` bindings is:

```rust
use metal_oxide::{Device, LaunchConfig};

mod kernels {
    include!(env!("METAL_OXIDE_BINDINGS"));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 257_u32;
    let device = Device::system_default()?;
    let kernels = kernels::load(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;

    let a = device.buffer_from_slice(&vec![1.0_f32; n as usize])?;
    let b = device.buffer_from_slice(&vec![2.0_f32; n as usize])?;
    let mut out = device.buffer_zeroed::<f32>(n as usize)?;
    let config = LaunchConfig::<256>::for_elements(n)?;

    // SAFETY: distinct n-element buffers and a 1D launch give one writer per index.
    unsafe {
        kernels.vec_add(config, &a, &b, &mut out, n)?;
    }

    assert!(out.as_slice().iter().all(|&value| value == 3.0));
    Ok(())
}
```

`LaunchConfig::<256>` sets 256 threads per block. For 257 elements,
`for_elements` produces two blocks, so 512 GPU threads execute. The kernel's
`i < n` condition limits writes to indices 0 through 256.

The launch configuration controls dispatch geometry. The scalar `n` controls
the kernel's own bounds condition. They are separate inputs. The wrapper takes
five values after `self`: the configuration and four kernel arguments.

When an entrypoint declares `#[kernel(block = (256, 1, 1))]`, its ABI records
that shape and the wrapper accepts the matching `LaunchConfig` type.
`vec_add` has no required shape, so its wrapper accepts const or dynamic
configuration. Host const generics alone do not specialize the shader.

The runtime performs these steps:

1. `Device::system_default` creates the GPU context and selects a native backend.
2. `Module::from_artifact` reads the manifest, validates its format and
   requirements, verifies the library bytes, and loads those bytes into Metal.
3. `Pipeline::new` selects `vec_add`, creates the native compute pipeline,
   and retains that entrypoint's ABI.
4. The generated `vec_add` wrapper constructs the four ordered arguments.
5. `Device::launch` calls `Device::submit`, which creates a `Batch`.
6. `Batch::launch` checks resource context, ABI types/access, block requirements,
   and dispatch limits, then encodes bindings and the dispatch.
7. The batch commits. `Device::launch` waits for GPU completion and reports
   command errors before returning.
8. The host reads the output buffer.

The runtime's `Module` holds a loaded Metal library and ABI metadata.
`metal_oxide_ir::Module` holds the compiler's program representation.

The runtime checks the call interface against the ABI. The kernel's documented
bounds, aliasing, race, and participation requirements remain the unsafe caller's
responsibility.
For example, passing `n = 257` with a shorter buffer is not made valid by
matching the `Buffer<f32>` parameter type.

## Classic Metal and Metal 4

The Rust-to-MSL build and artifact format are shared. Native API selection
happens when creating the `Device`. By default, macOS 26 and a GPU supporting
the Metal 4 family select Metal 4; otherwise the runtime selects classic Metal.

The selected variant controls pipeline creation and batch encoding:

| Operation | Classic Metal | Metal 4 |
| --- | --- | --- |
| Pipeline creation | Device compute-pipeline API | `MTL4Compiler` |
| Command encoding | `MTLComputeCommandEncoder` | `MTL4ComputeCommandEncoder` and command allocator |
| Arguments | `setBuffer` and copied `setBytes` scalars | GPU addresses in an immutable `MTL4ArgumentTable`; scalars in retained buffers |
| Resource residency | Classic resource management | Explicit residency set |
| Dispatch ordering | Buffer barriers between dispatches | Encoder barriers and a queue barrier |
| Submission | Command-buffer commit and completion callback | Command-queue commit and feedback callback |

Both paths retain native resources until GPU completion and expose the same
`Device`, `Pipeline`, `Batch`, and `Submission` API.
`METAL_OXIDE_BACKEND=classic` or `metal4` forces a path for verification;
forcing Metal 4 on an unsupported device returns an error.

## Implementation map

| Responsibility | Source |
| --- | --- |
| Entrypoint marker | [`metal-oxide-macros/src/lib.rs`](crates/metal-oxide-macros/src/lib.rs) |
| Rustc callbacks | [`compiler/src/driver.rs`](crates/metal-oxide-compiler/src/driver.rs) |
| Entries and concrete instances | [`compiler/src/collect.rs`](crates/metal-oxide-compiler/src/collect.rs), [`monomorphize.rs`](crates/metal-oxide-compiler/src/monomorphize.rs) |
| MIR import and intrinsic lowering | [`compiler/src/import/mod.rs`](crates/metal-oxide-compiler/src/import/mod.rs) |
| IR data and validation | [`ir/src/model.rs`](crates/metal-oxide-ir/src/model.rs), [`validate.rs`](crates/metal-oxide-ir/src/validate.rs) |
| MSL, ABI, and bindings | [`codegen/src/emit.rs`](crates/metal-oxide-codegen/src/emit.rs), [`abi.rs`](crates/metal-oxide-codegen/src/abi.rs), [`bindings.rs`](crates/metal-oxide-codegen/src/bindings.rs) |
| Compiler output files | [`compiler/src/output.rs`](crates/metal-oxide-compiler/src/output.rs) |
| Artifact model and validation | [`artifact/src/model.rs`](crates/metal-oxide-artifact/src/model.rs), [`validate.rs`](crates/metal-oxide-artifact/src/validate.rs) |
| Kernel build and Apple compiler | [`cargo-metal/src/rust.rs`](crates/cargo-metal/src/rust.rs), [`toolchain.rs`](crates/cargo-metal/src/toolchain.rs) |
| Manifest, cache, and host build | [`cargo-metal/src/build.rs`](crates/cargo-metal/src/build.rs), [`cache.rs`](crates/cargo-metal/src/cache.rs) |
| Library loading and launch validation | [`runtime/metal/module.rs`](crates/metal-oxide/src/metal/module.rs), [`pipeline.rs`](crates/metal-oxide/src/metal/pipeline.rs) |
| Submission and native paths | [`runtime/metal/device.rs`](crates/metal-oxide/src/metal/device.rs), [`batch.rs`](crates/metal-oxide/src/metal/batch.rs), [`backend.rs`](crates/metal-oxide/src/metal/backend.rs) |
