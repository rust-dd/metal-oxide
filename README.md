# metal-oxide

Rust compute kernels for Metal on Apple Silicon. Kernels are ordinary Rust functions in any
crate, beside the host code that launches them, with CUDA-style thread/block indices and a
stable Rust host runtime. Supports classic Metal and Metal 4.

## Example

```rust
use metal_oxide::device::{DisjointBuffer, ReadBuffer, thread_index};
use metal_oxide::{Device, LaunchConfig, Operation};

#[metal_oxide::kernel(requires(a.len() == out.len(), b.len() == out.len()))]
pub fn vec_add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    if let Some(out) = out.get_mut(&index) {
        let i = index.get();
        *out = a.as_slice()[i] + b.as_slice()[i];
    }
}

fn main() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let a = device.buffer_from_slice(&[1.0_f32, 2.0, 3.0])?;
    let b = device.buffer_from_slice(&[10.0_f32, 20.0, 30.0])?;
    let mut out = device.buffer_zeroed::<f32>(3)?;
    let config = LaunchConfig::<256>::for_elements(3)?;

    vec_add(&device, config, &a, &b, &mut out).wait()?;

    assert_eq!(out.read().wait(), &[11.0, 22.0, 33.0]);
    Ok(())
}
```

`cargo build` compiles the kernel for Metal and replaces it with a function of the same name
that launches it. Slices carry the lengths the host bound, so their indexing is checked.
`get_mut` gives each thread `&mut` to its own element only, and `requires` lists relations
every launch checks before it encodes (`Error::Contract` otherwise); the compiler assumes
them, so here no check remains. A safe `fn` kernel launches safely; the launch function of
an `unsafe fn` kernel, such as one using unchecked loads or barriers, is `unsafe` too, and
its callers uphold the contract in its doc comment.

A launch returns a lazy operation: nothing runs until it is awaited or waited. Arguments come
back in the form they were passed, so owned and `Arc` buffers make `'static` futures for
`tokio::spawn`, and `buffer.read().await` waits for the GPU without blocking the executor
thread:

```rust
let (a, b, out) = vec_add(&device, config, a, b, out).await?;
let values = out.read().await;
```

`metal_oxide::sequence(|batch| ...)` runs the operations it pushes in one command buffer: a
launch waits for the earlier ones whose buffers it uses, and the others run concurrently;
`operation.timed()` also returns the GPU time.

`then` continues an operation once it has succeeded, with dependent launches or async
readback, and `map` processes its result on the CPU. An error skips the rest of the chain,
and dropping a chain starts none of its later stages:

```rust
let mean = vec_add(&device, config, &a, &b, &mut out)
    .then(|(_, _, out)| async move { Ok(out.read().await.iter().sum::<f32>()) })
    .map(|sum| sum / 3.0)
    .await?;
```

Within a sequence, recorded buffers can feed later launches. A CPU read panics if the
buffer has a recorded write; CPU writes, upload and native export panic while any access
is recorded. Conflicting nested batches return `InvalidBuffer`.

First-use generated pipelines compile on a worker. Awaiting a launch yields while its
pipeline or command storage is unavailable; `wait()` blocks through both preparation
and execution.

Kernels that share types and helpers go in a `#[kernels]` module. The host uses the same
types; the module's `#[repr(C)]` records that kernels take become `GpuValue`s:

```rust
#[metal_oxide::kernels(math = "fast", dependencies(libm))]
pub mod particles {
    use metal_oxide::device::{DisjointBuffer, thread_index};

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Particle { pub position: [f32; 2], pub velocity: [f32; 2] }

    #[kernel]
    pub fn step(mut particles: DisjointBuffer<Particle>, dt: f32) { /* ... */ }
}

particles::step(&device, config, &mut state, 0.01).await?;
```

`dependencies(...)` names the `no_std` crates the kernels use. A large kernel code base can
stay a `no_std` crate of its own: `#[metal_oxide::kernels(path = "../kernels")] mod kernels {}`
launches its kernels with a host struct for each record they take. See the
[examples](https://github.com/rust-dd/metal-oxide/tree/main/examples).

Each build saves its artifact beside the crate's sources, in
`metal/<file>/<item>-<math mode>/`. Commit and publish that directory: without the kernel
toolchain, and for the crate's dependents, the saved artifact stands in while the kernels
are unchanged. `METAL_OXIDE_ARTIFACTS=locked` builds from saved artifacts only and turns a
stale one into an error, for CI. rust-analyzer expands the macros without a kernel build.

Use `Device::all()?.into_iter().nth(index)` to select a device explicitly.
`recommended_max_working_set_size()` reports its advisory memory budget.
Devices, modules, pipelines, buffers and operations are `Send + Sync`; `Device` clones share
its command queues and its loaded kernels: an operation runs after the earlier operations
that use its buffers, independent ones run on other queues and finish independently, and CPU
views on any thread wait for pending GPU work. Native Metal integration uses
`Device::from_native`, unsafe `Buffer::from_native`, and consuming `Buffer::into_native`.
See the [ownership contract](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md#native-resource-ownership).

Checks the compiler cannot prove run on the GPU: an out-of-bounds index, an overflow
under the profile's overflow checks, a division by zero, an `unwrap` or `expect` of `None`
or `Err`, or a literal `assert!`/`panic!` stops the thread and returns `Error::KernelFailed`
with the kernel, dispatch, Rust location and message; the dispatches that wait for it in its
command buffer do not run. An operation committed while it ran that reads its writes skips
its generated kernels and returns `Error::DependencyFailed` with the same failure. A check
before a barrier or SIMD collective must be proved or uniform. `metal_oxide::device::log!`
prints from kernels to stderr through Metal shader logging.

Generic kernels list their concrete type and const arguments explicitly:
`#[kernel(instances(scale_f32(f32, 2), scale_u32(u32, 3)))]` on
`fn scale<T, const N: usize>(...)` produces `scale_f32` and `scale_u32` launch functions.
Each instance keeps the function's safety, `block` shape and `requires` relations.
This works in host functions, inline modules and separate kernel crates.
Host signatures spell associated types as `<T as Trait>::Type`.

Lifetime parameters are checked by Rust and erased before GPU code generation;
they need no instance arguments. References and lifetime-bearing structs work inside
kernels and helpers. Only owned values and buffer handles cross the kernel boundary.

Kernels, and the `no_std` crates that `dependencies(...)` names, compile from rustc's MIR
against `core`: references and slices, records, tuples and arrays, enums with payloads,
`Option` and `Result` with `?`, closures, core's iterators and generic helpers, over `bool`,
`f32`, `F16`, `BF16`, the 8- to 32-bit integers, `u64`, `usize` and `isize`. The compiler rejects
what it cannot lower with a diagnostic at the Rust source, such as recursion, calls through
function pointers or trait objects, destructors, coroutines, `f64`,
`i64` and 128-bit integers.

`F16` and `BF16` support rounded arithmetic and f32 conversions; the runtime exports
their host counterparts from `half`. Separate kernel crates can also use Rust's
unstable `f16` primitive. Bfloat kernels record a GPU requirement and reject unsupported
devices when loaded. Conversions preserve finite subnormal values; arithmetic follows
the selected [numerical mode](CONCEPT.md#apple-compilation-and-artifact).

Within a thread, `core::simd::Simd<T, N>` supports 2–4 lanes of `f32` and 8-, 16- or
32-bit integers. Construction, lane access, arithmetic, masks, comparisons, selection,
reductions, casts, swizzles, saturation and bit counts lower to scalar operations.
Integer division retains zero checks and wrapping `MIN / -1` behavior. Vectors cannot
cross the kernel boundary; wider vectors, gather/scatter and masked loads are rejected.
Host files using `core::simd` need nightly Rust; separate kernel crates use the pinned
compiler toolchain.

The `no_std` integration tests compile and execute `libm` f32 math and `num-traits`
generic float/integer workloads. Dependencies remain subject to the supported device
types and operations; using `no_std` alone does not make an entire crate compatible.

Kernels can request runtime-sized block-local storage with
`DynamicThreadgroupBuffer<T>`. Launch functions take an element count; raw `Launch`
operations use `Argument::threadgroup::<T>(elements)`. Both paths validate combined
static/dynamic memory against `Device::max_threadgroup_memory_per_block()`.
The [spectral example](https://github.com/rust-dd/metal-oxide/tree/main/examples/spectral) uses this for runtime-sized FFT tiles,
with CPU/native references, seeded stream checks and matched-mode benchmarks.

Device atomics take a constant `core::sync::atomic::Ordering`. `u32`/`i32` support
load, store, exchange, weak compare/exchange, arithmetic, minimum/maximum and bitwise
operations in device and threadgroup memory. `f32` supports device load, store,
exchange, weak compare/exchange and add/subtract. Its compare/exchange compares bits.
`u64` supports device `min`/`max` without returning the previous value. Unsupported
operations and `SeqCst` are rejected. Acquire/release requires MSL 4.1 and macOS 27;
its lowering and capability gates are tested, but hardware validation is pending.

Artifacts carry these requirements; the runtime checks OS and GPU support before
loading them. The compiler baseline is MSL 3.2 and macOS 15, even when a feature's
minimum is lower. Regenerate this table with
`cargo run -p metal-oxide-artifact --example capabilities`.

<!-- capabilities -->
| Artifact feature | MSL | macOS | Apple GPU family | Mac2 |
| --- | --- | --- | --- | --- |
| `float32_atomics` | 3.0 | 13 | 7 | no |
| `uint64_atomics` | 2.4 | 12 | 8 | no |
| `atomic_acquire_release` | 4.1 | 27 | 7 | no |
| `int32_atomics` | 1.0 | 11 | 1 | yes |
| `simd_groups` | 2.0 | 11 | 7 | yes |
| `dynamic_threadgroup_memory` | 1.0 | 11 | 1 | yes |
| `logging` | 3.2 | 15 | 6 | yes |
| `bfloat16` | 3.1 | 14 | 6 | no |
<!-- /capabilities -->

Requirements follow Apple's [Metal language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
and [feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf).
The `uint64_atomics` feature covers only the minimum/maximum operations supported on
Apple family 8 and later on macOS.

## Compiler flow

```text
#[kernel] / #[kernels] item
    |
rustc MIR -> metal-oxide IR
                 +-> MSL -> Apple compiler -> .metallib
                 +-> ABI -> launch functions          |
                                |                     |
                           Rust host + runtime <------+
```

## Run

The runtime supports Rust 1.89+, macOS 15+ on Apple Silicon. Building kernels also
requires the pinned nightly with `rustc-dev`, `rust-src` and `llvm-tools`, and Xcode with the
Metal Toolchain; `cargo metal setup` installs them. Development uses the workspace's Rust
1.99 toolchain. From the repository root:

```sh
cargo run -p cargo-metal -- setup
cargo run -p vec-add
```

Kernels build with the metal-oxide compiler from the sources of the `metal-oxide` they
depend on, as in this checkout or a git dependency.

`scripts/test-gpu.sh` passes on:

| GPU | macOS | Xcode | Native APIs |
| --- | --- | --- | --- |
| M4 Max | 26.2 | 26.6 | classic Metal, Metal 4 |
| Apple Paravirtual device, GitHub-hosted M1 VM (CI) | 15.7, 26.6 | 16.4, 26.6 | classic Metal |

Beyond these suites, seeded generated programs run on the CPU through rustc, as their MSL
built for the host, and on the GPU: every thread's outputs must agree, and a failed check
must be one the CPU panicked at, with its Rust location. Failing programs are shrunk and
kept as regression fixtures, and a workflow fuzzes nightly on hosted classic Metal. Each
compiler rejection is either reached by a fixture at its expected location or documented as
unreachable from kernel source. The generated MSL includes workarounds for Apple compiler
defects found by fuzzing; see [MSL](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md#msl).

Math defaults to precise functions without implicit floating-point contraction.
`#[kernels(math = "fast")]` opts into fast functions, reassociation, reciprocal
approximations and contraction. Arithmetic keeps NaN and infinity support; fast functions
have narrower accuracy/domain guarantees and zero signs are unspecified. The mode is
recorded in the artifact; see [numerical compilation](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md#apple-compilation-and-artifact).

`cargo metal run|test --validate` enables Metal API and shader validation,
`--capture FILE.gputrace` records a GPU trace that Xcode shows with Rust lines, and
`cargo metal inspect --emit ir|msl|air` prints what the kernels compiled to; see
[debugging](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md#debugging).

## Compared with other Rust GPU projects

As of October 2026:

| Project | Kernels | Compiled | Targets | On Apple GPUs |
| --- | --- | --- | --- | --- |
| metal-oxide | Rust functions | from rustc MIR to MSL at `cargo build`; saved artifacts | Apple Silicon | native Metal and Metal 4 |
| [CubeCL](https://github.com/tracel-ai/cubecl) | `#[cube]` Rust functions, which a macro rewrites to build its IR | JIT at first launch, with a cache | NVIDIA, AMD, Apple, Vulkan, WebGPU, CPU | MSL through wgpu or its Metal runtime |
| [Burn](https://github.com/tracel-ai/burn) | tensor operations; custom kernels in CubeCL | through CubeCL | CubeCL's | through CubeCL |
| [Candle](https://github.com/huggingface/candle) | tensor operations | Metal kernels from MSL source at run time | CPU, NVIDIA, Apple | hand-written MSL |
| [Rust GPU](https://github.com/Rust-GPU/rust-gpu) | `no_std` Rust | rustc backend to SPIR-V, in a build script | Vulkan; others through naga | SPIR-V translated to MSL, through wgpu or MoltenVK |
| [CUDA Rust](https://github.com/NVIDIA/cuda-rust) (NVIDIA) | Rust (cuda-oxide) or a tile DSL (cuTile Rust) | rustc backend to PTX; tile kernels JIT | NVIDIA on Linux | no |
| [Rust CUDA](https://github.com/Rust-GPU/rust-cuda) | `no_std` Rust | rustc backend to NVVM IR and PTX, in a build script | NVIDIA | no |
| [wgpu](https://github.com/gfx-rs/wgpu) | WGSL | naga, at run time | Vulkan, Metal, D3D12, OpenGL, WebGPU | WGSL translated to MSL |
| [objc2-metal](https://github.com/madsmtm/objc2) | MSL | Apple's compiler | Apple | bindings to the Metal API |

metal-oxide is not yet released. Depend on the git repository; the API may change before
the first release.

[Architecture](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md) · [Build, test, benchmarks](https://github.com/rust-dd/metal-oxide/blob/main/CONTRIBUTING.md) · [MIT](https://github.com/rust-dd/metal-oxide/blob/main/LICENSE)
