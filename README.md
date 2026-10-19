# metal-oxide

Compile Rust compute kernels to Metal on Apple Silicon. Kernels use CUDA-style
thread and block coordinates. The host runtime supports stable Rust and provides
one API for classic Metal and Metal 4.

There is no registry release yet. Use a path or git dependency; the API may change.
The [book](book/src/SUMMARY.md) covers installation, kernel authoring and debugging.

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

`cargo build` compiles the kernel through rustc MIR and MSL into a `.metallib`.
The macro generates a host launch function with the same name. Launches are lazy:
execution starts when the operation is awaited or waited.

`DisjointBuffer` gives each thread mutable access to its own element. `requires`
checks the length relations before launch, allowing the compiler to prove the
indexing safe. Checks that remain run on the GPU and report `Error::KernelFailed`
with the Rust source location. An `unsafe fn` kernel produces an unsafe launch
function whose caller must uphold its documented contract.

## Run the checkout

GPU execution requires macOS 15 or later on Apple Silicon. The host runtime
supports Rust 1.89 or later; this checkout uses Rust 1.99. Kernel authoring also
requires Xcode with its Metal Toolchain and the pinned nightly Rust compiler.
Install Xcode and select its developer directory, then run:

```sh
git clone https://github.com/rust-dd/metal-oxide.git
cd metal-oxide
cargo run -p cargo-metal --locked -- setup
cargo run -p vec-add --locked
```

Setup installs the pinned Rust components and downloads the Metal Toolchain if
missing. For use in another application, follow the
[installation guide](book/src/installation.md).

Builds save kernels under `metal/<file>/<item>-<math mode>/`. Commit and distribute
these artifacts with the crate. Unchanged kernels can then be used without the
kernel compiler. `METAL_OXIDE_ARTIFACTS=locked` requires current saved artifacts and
fails if they are stale. rust-analyzer expands the macros without building kernels.

## Kernel and runtime APIs

- Write a standalone `#[kernel]`, group shared types and helpers in a `#[kernels]`
  module, or keep kernels in a separate `no_std` crate. Generic entrypoints declare
  their concrete type and const instances. See [supported Rust](book/src/supported-rust.md).
- `LaunchConfig<X, Y, Z>` fixes the threads per block; its `grid` counts blocks.
  `DynamicLaunchConfig` accepts a runtime block shape. Both validate device limits.
  See [launch geometry](book/src/geometry.md).
- Buffers support borrowed and owned arguments; `Arc` arguments are read-only.
  Await readback with `buffer.read().await`; use `sequence` for several operations
  in one command buffer. See [memory](book/src/memory.md) and
  [async operations](book/src/async.md).
- Kernels can use threadgroup memory, SIMD-group operations and 8×8 matrix
  operations, subject to the device requirements below.
- Precise math is the default. `#[kernels(math = "fast")]` permits fast functions,
  reassociation and contraction. See [numerical modes](book/src/performance.md#numerical-modes).
- `cargo metal run|test --validate` enables Metal validation.
  `cargo metal inspect --emit ir|msl|air` prints compiler output.
  `cargo metal run|test --capture FILE.gputrace` records a GPU trace.
  See [debugging](book/src/debugging.md).

## Known limitations

- **Unresolved numerical failure:** a scheduled classic Metal fuzz run on macOS 15
  produced a different finite result from CPU Rust and host-executed MSL. The
  minimized case combines narrow integer bit reversal and float arithmetic; its
  cause is still under investigation. See the
  [failing run](https://github.com/rust-dd/metal-oxide/actions/runs/39218928471).
- **Hardware coverage:** native M1–M3 GPUs and a second Metal 4 GPU generation have
  not been verified. Hosted Paravirtual GPUs do not establish native M1 coverage.
- **Rust subset:** heap allocation, host I/O, recursion, indirect calls,
  destructors, coroutines, `f64`, `i64` and 128-bit integers are unsupported.
  A dependency being `no_std` does not make all its operations usable in kernels.
  Device crates and their dependencies cannot use build scripts.
- **Vectors and matrices:** `core::simd` supports 2–4 lanes of f32 or 8/16/32-bit
  integers. Wider vectors, gather/scatter and masked loads are rejected. Vectors
  cannot cross the kernel boundary. SIMD-group matrices require uniform operands
  and control, and are unavailable on the hosted Paravirtual device.
- **Atomics:** supported operations depend on type and address space. `SeqCst` is
  rejected. Acquire/release requires macOS 27 and MSL 4.1; its hardware validation
  is pending.
- **Build configuration:** Cargo configuration `include` directives are
  unsupported. Adding a previously absent configuration file that changes only
  the release profile requires touching a kernel source or running `cargo clean`.
  See [saved artifacts](book/src/installation.md#saved-artifacts).
- **Tooling validation:** automated rust-analyzer checks pass; manual editor
  rendering and Xcode Rust source display remain unverified.

## Hardware coverage

Hardware suites have run on these configurations:

| GPU | macOS | Xcode | Native APIs |
| --- | --- | --- | --- |
| M4 Max | 26.2 | 26.6 | classic Metal, Metal 4 |
| Apple Paravirtual device, GitHub-hosted M1 VM | 15.7, 26.6 | 16.4, 26.6 | classic Metal |

Differential tests compare generated programs on CPU Rust, host-executed MSL and
the GPU. Scheduled fuzzing runs on hosted classic Metal; failing programs are
minimized and uploaded as CI artifacts. Retained regression fixtures run in the
regular test suites. See
[verification and benchmarks](CONTRIBUTING.md) for commands and recorded results.

## Device requirements

Artifacts record their feature requirements; the runtime checks them before
loading. The compiler baseline is MSL 3.2 and macOS 15, even when a feature's
minimum is lower. These requirements describe device eligibility, not test coverage.

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

The `uint64_atomics` feature covers device min/max without returning the previous
value. Regenerate the table with
`cargo run -p metal-oxide-artifact --example capabilities --locked`.

[Architecture](CONCEPT.md) · [Build and test](CONTRIBUTING.md) · [Examples](examples) · [MIT](LICENSE)
