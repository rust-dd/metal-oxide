# metal-oxide

Write compute kernels in Rust and run them on Apple GPUs. metal-oxide compiles
Rust functions through rustc MIR to Metal Shading Language, then uses Apple's
compiler to produce a `.metallib`. Each kernel gets a typed Rust launch function.

Kernels can live beside the host code that calls them, in a shared module, or in
a separate `no_std` crate. The runtime supports stable Rust, async execution and
both classic Metal and Metal 4 through the same API.

**Early alpha.** The API may change between releases.
See [known limitations](#known-limitations) for language
restrictions and hardware coverage.

## A kernel and its host

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

`cargo build` compiles the kernel and generates its launch function. No application
build script is needed. `DisjointBuffer` gives each thread mutable access to its
own element; `requires` checks the buffer-length relations before launch so the
compiler can prove the indexing safe.

Checks the compiler cannot prove run on the GPU. Bounds failures, division by
zero and assertions return `Error::KernelFailed` with the Rust location. Kernels
always build with the release profile, even when the host uses a debug build.
Integer overflow checks are off by default; set `overflow-checks = true` under
`[profile.release]` in the workspace `Cargo.toml` to report overflows as kernel
failures too. A kernel declared `unsafe fn` generates an unsafe launch function;
callers must uphold the kernel's documented contract.

## Launches, buffers and async work

`LaunchConfig<X, Y, Z>` specifies threads per block; its `grid` counts blocks.
`DynamicLaunchConfig` accepts a runtime block shape. Kernels read CUDA-style
`thread_idx()`, `block_idx()`, `block_dim()` and `grid_dim()` coordinates. Each
block maps to one Metal threadgroup.

A launch is lazy: calling it creates an operation, and awaiting it or calling
`.wait()` drives execution. The same kernel can be used from async code:

```rust
let (a, b, out) = vec_add(&device, config, a, b, out).await?;
let values = out.read().await;
```

Arguments come back in the form they were passed. Owned buffers allow a launch
to move into a task; shared reads also accept `Arc<Buffer<T>>`. Writes require an
owned buffer or an exclusive borrow. GPU readback can be awaited without blocking
the executor thread.

`sequence` records several operations in one command buffer. The runtime orders
conflicting buffer accesses and allows independent work to overlap. `then` chains
dependent work, `map` processes its result on the CPU, and `operation.timed()`
returns optional GPU timing with the result. Native Metal resources can be imported and
exported under the [ownership contract](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md#native-resource-ownership).

## Kernel code

The compiler works with a subset of Rust, checked by rustc before GPU lowering:

- References, slices, records, tuples, arrays, enums, `Option`, `Result`, closures,
  core iterators and generic helpers. Generic entrypoints list their concrete
  type and const instances.
- `f32`, `F16`, `BF16`, 8/16/32-bit integers and `u64`. Supported `libm` and
  `num-traits` workloads are exercised by integration tests.
- Static and dynamic threadgroup memory, barriers, SIMD-group reductions,
  shuffles and scans, plus 8×8 SIMD-group matrix operations.
- Device and threadgroup atomics, with supported operations and orderings checked
  during compilation. Feature requirements are recorded in each artifact.

Use `#[metal_oxide::kernels]` to share types and helpers between kernels. A
`#[kernels(path = "../kernels")]` module loads a separate kernel crate; its
`no_std` dependencies compile for the device alongside it.

Precise math is the default and disables implicit floating-point contraction.
`#[kernels(math = "fast")]` permits fast functions, reassociation and contraction.
The selected mode is recorded in the artifact. See
[supported Rust](https://rust-dd.github.io/metal-oxide/supported-rust.html) and
[numerical modes](https://rust-dd.github.io/metal-oxide/performance.html#numerical-modes) for the exact boundaries.

## Compared with other Rust GPU projects

These projects operate at different levels: kernel compilers, tensor libraries,
portable GPU APIs and native bindings.

| Project | Kernel authoring | Compilation | Targets | Apple GPU path |
| --- | --- | --- | --- | --- |
| **metal-oxide** | Rust functions | rustc MIR → MSL at build time; saved artifacts | Apple Silicon on macOS | Native Metal and Metal 4 |
| [CubeCL](https://github.com/tracel-ai/cubecl) | `#[cube]` Rust functions rewritten into IR-building code | JIT with caching | CUDA, HIP, Metal, Vulkan, WebGPU, CPU | MSL through wgpu or `metal-native` |
| [Burn](https://github.com/tracel-ai/burn#supported-backends) | Tensor operations; custom CubeCL kernels | CubeCL JIT for accelerated backends; eager CPU backends also available | GPUs through CubeCL; CPU and Wasm backends | CubeCL Metal/WebGPU backends |
| [Candle](https://github.com/huggingface/candle) | Tensor operations backed by device kernels | Embedded MSL compiled and cached at runtime | CPU, CUDA, Metal | Handwritten MSL |
| [Rust GPU](https://github.com/Rust-GPU/rust-gpu) | `no_std` Rust shaders | rustc backend → SPIR-V; `spirv-builder` or `cargo gpu` | Vulkan and other SPIR-V consumers | Translation through compatible wgpu/MoltenVK paths |
| [CUDA Rust — NVIDIA](https://github.com/NVIDIA/cuda-rust) | Rust SIMT kernels (`cuda-oxide`) or tile kernels (`cutile-rs`) | SIMT → PTX ahead of time; tiles → Tile IR → cubin via JIT | NVIDIA CUDA | — |
| [Rust CUDA](https://github.com/Rust-GPU/rust-cuda) | `no_std` Rust | rustc/NVVM backend → PTX | NVIDIA CUDA | — |
| [wgpu](https://github.com/gfx-rs/wgpu) | WGSL; optional SPIR-V, GLSL or Naga IR | Naga translation and native shader compilation at runtime | Vulkan, Metal, D3D12, OpenGL, WebGPU/WebGL2 | Naga → MSL, native Metal backend |
| [objc2-metal](https://github.com/madsmtm/objc2) | Metal shaders or precompiled libraries | Apple's compiler and Metal APIs | Apple platforms | Native bindings; no Rust kernel compiler |

## Try it

GPU execution requires **macOS 15+ on Apple Silicon**. The host runtime supports
**Rust 1.89+**; development uses Rust 1.99. Building kernels also requires Xcode
with its Metal Toolchain and the pinned nightly Rust compiler. Install Xcode and
select its developer directory. In an existing Rust application, run:

```sh
cargo install cargo-metal --version '=0.1.0-alpha.1' --locked
cargo metal setup
cargo add metal-oxide@=0.1.0-alpha.1
```

Copy the kernel example above into `src/main.rs`, then run `cargo run`.
Setup installs the pinned Rust components and downloads the Metal Toolchain if
missing. Specify the alpha version when installing; plain `cargo install`
skips prereleases.

To run the repository's examples:

```sh
git clone --branch v0.1.0-alpha.1 https://github.com/rust-dd/metal-oxide.git
cd metal-oxide
cargo run -p cargo-metal --locked -- setup
cargo run -p vec-add --locked
```

Keep the dependency and `cargo-metal` at the same version. The
[installation guide](https://rust-dd.github.io/metal-oxide/installation.html) also covers development checkouts
and saved artifacts.

Kernel builds save `metal/<file>/<item>-<math mode>/` beside the crate's sources.
Commit and distribute that directory: dependents can use unchanged kernels
without installing the kernel compiler. `METAL_OXIDE_ARTIFACTS=locked` uses saved
artifacts only and rejects stale ones.

## Debugging and verification

- `cargo metal run|test --validate` enables Metal API and shader validation.
- `cargo metal inspect --emit ir|msl|air` prints the generated code.
- `cargo metal run|test --capture FILE.gputrace` records a GPU trace for Xcode.
- rust-analyzer expands kernel macros without invoking the kernel compiler.

Differential tests compare CPU Rust, host-executed MSL and GPU results. Scheduled
fuzzing minimizes failures and uploads them as CI artifacts; retained regression
fixtures run in the regular test suites. Benchmarks compare generated kernels
with CPU references and handwritten MSL, with MPS, MPP and MLX comparisons where
applicable.

Hardware suites have run on these configurations:

| GPU | macOS | Xcode | Native APIs |
| --- | --- | --- | --- |
| M4 Max | 26.2 | 26.6 | classic Metal, Metal 4 |
| Apple Paravirtual device, GitHub-hosted M1 VM | 15.7, 26.6 | 16.4, 26.6 | classic Metal |

The [example index](https://github.com/rust-dd/metal-oxide/blob/main/examples/README.md) lists workloads and run commands.
The [contribution guide](https://github.com/rust-dd/metal-oxide/blob/main/CONTRIBUTING.md) has test commands and benchmark results.
The [book](https://rust-dd.github.io/metal-oxide/) covers kernel authoring, memory, async execution,
performance and debugging.

## Known limitations

- **Hardware coverage is incomplete.** Native M1–M3 GPUs and a second Metal 4 GPU
  generation have not been verified. Paravirtual execution does not establish
  native M1 coverage.
- **Not all Rust is device code.** Heap allocation, host I/O, recursion, indirect
  calls, destructors, coroutines, `f64`, `i64` and 128-bit integers are unsupported.
  Device dependencies cannot use build scripts; `no_std` alone does not establish
  compatibility.
- **Cooperation has restrictions.** Barriers and SIMD collectives require uniform
  participation. SIMD-group matrices require uniform operands and control.
  Within-thread `core::simd` supports only 2–4 lanes of f32 or 8/16/32-bit integers;
  wider vectors, gather/scatter and masked loads are rejected.
- **Atomics are limited by type and address space.** `SeqCst` is rejected.
  Acquire/release needs macOS 27/MSL 4.1 and still awaits hardware validation.
- **Cargo configuration has two gaps.** Configuration `include` directives are
  unsupported. After adding a new configuration file that changes only the
  release profile, touch a kernel source or run `cargo clean`.
- **Manual tooling checks remain open.** Automated rust-analyzer checks pass;
  editor rendering and Rust source display in Xcode have not been verified.

## Device feature requirements

The runtime checks artifact requirements before loading. The compiler baseline
is MSL 3.2 and macOS 15 even when a feature's minimum is lower. This table describes
eligibility, not hardware test coverage.

<!-- capabilities -->
| Artifact feature | MSL | macOS | Apple GPU family | Mac2 |
| --- | --- | --- | --- | --- |
| `float32_atomics` | 3.0 | 13 | 7 | no |
| `uint64_atomics` | 2.4 | 12 | 8 | no |
| `atomic_acquire_release` | 4.1 | 27 | 7 | no |
| `int32_atomics` | 1.0 | 11 | 1 | yes |
| `simd_groups` | 2.0 | 11 | 7 | yes |
| `simdgroup_matrix` | 2.3 | 11 | 7 | no |
| `dynamic_threadgroup_memory` | 1.0 | 11 | 1 | yes |
| `logging` | 3.2 | 15 | 6 | yes |
| `bfloat16` | 3.1 | 14 | 6 | no |
<!-- /capabilities -->

`uint64_atomics` covers device min/max without returning the previous value.
Regenerate this table with
`cargo run -p metal-oxide-artifact --example capabilities --locked`.

[Architecture](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md) · [Examples](https://github.com/rust-dd/metal-oxide/tree/main/examples) · [Contributing](https://github.com/rust-dd/metal-oxide/blob/main/CONTRIBUTING.md) · [MIT license](https://github.com/rust-dd/metal-oxide/blob/main/LICENSE)
