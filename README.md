# metal-oxide

Rust compute kernels for Metal on Apple Silicon. Ordinary `no_std` Rust kernels,
CUDA-style thread/block indices, and a stable Rust host runtime. Supports classic
Metal and Metal 4.

## Example

Kernel (`kernels/src/lib.rs`):

```rust
#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

/// # Safety
/// Nothing writes `a` or `b` while the kernel runs.
#[kernel]
pub unsafe fn vec_add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
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

Slices carry the lengths the host bound, so their indexing is checked; here the guards
prove it and no check remains. `get_mut` gives each thread `&mut` to its own element only.

Host (`host/src/main.rs`):

```rust
use metal_oxide::{Device, LaunchConfig, Operation};

mod kernels {
    include!(env!("METAL_OXIDE_BINDINGS"));
}

fn main() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let kernels = kernels::load(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
    let a = device.buffer_from_slice(&[1.0_f32, 2.0, 3.0])?;
    let b = device.buffer_from_slice(&[10.0_f32, 20.0, 30.0])?;
    let mut out = device.buffer_zeroed::<f32>(3)?;
    let config = LaunchConfig::<256>::for_elements(3)?;

    // SAFETY: nothing else writes the inputs while the kernel runs.
    unsafe { kernels.vec_add(config, &a, &b, &mut out) }.wait()?;

    assert_eq!(out.read().wait(), &[11.0, 22.0, 33.0]);
    Ok(())
}
```

A kernel method returns a lazy operation: nothing runs until it is awaited or waited.
Arguments come back in the form they were passed, so owned and `Arc` buffers make
`'static` futures for `tokio::spawn`, and `buffer.read().await` waits for the GPU without
blocking the executor thread:

```rust
let (a, b, out) = unsafe { kernels.vec_add(config, a, b, out) }.await?;
let values = out.read().await;
```

`metal_oxide::sequence(|batch| ...)` runs the operations it pushes in one command buffer: a
launch waits for the earlier ones whose buffers it uses, and the others run concurrently;
`operation.timed()` also returns the GPU time.

`cargo-metal` generates the bindings and supplies the build paths.
Complete Cargo setup: [examples/vec-add](https://github.com/rust-dd/metal-oxide/tree/main/examples/vec-add).

Hosts that publish their kernels keep the artifact in their sources and let a
build script with the `metal-oxide-build` build dependency maintain it:

```rust
// build.rs
fn main() -> Result<(), metal_oxide_build::Error> {
    metal_oxide_build::Artifact::new("../kernels/Cargo.toml", "src/kernels").update()
}
```

`cargo build` rebuilds `src/kernels` when the kernel sources or the metal-oxide
revision change, and records its inputs in `inputs.json`. Otherwise, and in a
published package without the kernel crate, it uses the committed files and needs
neither the nightly nor the Metal toolchain. `METAL_OXIDE_ARTIFACTS=locked` turns a
stale artifact into an error for CI. The host embeds the files with
`kernels::load_bytes(&device, include_str!("kernels/manifest.json"),
include_bytes!("kernels/kernels.metallib"))`; pipelines are created on first use.
See [generated bindings](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md#generated-rust-bindings).

Use `Device::all()?.into_iter().nth(index)` to select a device explicitly.
`recommended_max_working_set_size()` reports its advisory memory budget.
Devices, modules, pipelines, buffers, operations and generated `Kernels` are `Send + Sync`;
`Device` clones share its command queues: an operation runs after the earlier operations
that use its buffers, independent ones run on other queues and finish independently, and CPU
views on any thread wait for pending GPU work. Native Metal integration uses
`Device::from_native`, unsafe `Buffer::from_native`, and consuming `Buffer::into_native`.
See the [ownership contract](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md#native-resource-ownership).

Checks the compiler cannot prove run on the GPU: an out-of-bounds index, an overflow
under the profile's overflow checks, a division by zero, an `unwrap` or `expect` of `None`
or `Err`, or a literal `assert!`/`panic!` stops the thread and returns `Error::KernelFailed`
with the kernel, dispatch, Rust location and message; the dispatches that wait for it in its command buffer do not run. An
operation committed while it ran that reads its writes skips its generated kernels and
returns `Error::DependencyFailed` with the same failure. A check before a barrier or SIMD
collective must be proved or uniform. `metal_oxide_device::log!` prints from kernels to
stderr through Metal shader logging.

Kernels can request runtime-sized block-local storage with
`DynamicThreadgroupBuffer<T>`. Generated bindings take an element count;
raw `Launch` operations use `Argument::threadgroup::<T>(elements)`. Both paths validate
combined static/dynamic memory against `Device::max_threadgroup_memory_per_block()`.
The [spectral example](https://github.com/rust-dd/metal-oxide/tree/main/examples/spectral) uses this for runtime-sized FFT tiles,
with CPU/native references, seeded stream checks and matched-mode benchmarks.

## Compiler flow

```text
Rust kernel
    |
rustc MIR -> metal-oxide IR
                 +-> MSL -> Apple compiler -> .metallib
                 +-> ABI -> Rust bindings          |
                                |                 |
                           Rust host + runtime <--+
```

## Run

The runtime supports Rust 1.89+, macOS 15+ on Apple Silicon. Kernel compilation
also requires the pinned nightly and Xcode with the Metal Toolchain. Development
uses the workspace's Rust 1.99 toolchain.
From the repository root:

```sh
rustup toolchain install nightly-2026-10-04 --component rustc-dev --component rust-src --component llvm-tools
xcodebuild -downloadComponent MetalToolchain
cargo run -p cargo-metal -- doctor
cargo run -p cargo-metal -- run -p vec-add
```

`scripts/test-gpu.sh` passes on:

| GPU | macOS | Xcode | Native APIs |
| --- | --- | --- | --- |
| M4 Max | 26.2 | 26.6 | classic Metal, Metal 4 |
| Apple Paravirtual device, GitHub-hosted M1 VM (CI) | 15.7, 26.6 | 16.4, 26.6 | classic Metal |

Math defaults to precise functions without implicit floating-point contraction.
Use `cargo metal build -p HOST --math-mode fast` to opt into fast functions,
reassociation, reciprocal approximations and contraction. Arithmetic keeps NaN
and infinity support; fast functions have narrower accuracy/domain guarantees
and zero signs are unspecified. The mode is recorded in the artifact and cache;
see [numerical compilation](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md#apple-compilation-and-artifact).

[Architecture](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md) · [Build, test, benchmarks](https://github.com/rust-dd/metal-oxide/blob/main/CONTRIBUTING.md) · [MIT](https://github.com/rust-dd/metal-oxide/blob/main/LICENSE)
