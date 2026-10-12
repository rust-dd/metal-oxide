# metal-oxide

Rust compute kernels for Metal on Apple Silicon. Ordinary `no_std` Rust kernels,
CUDA-style thread/block indices, and a stable Rust host runtime. Supports classic
Metal and Metal 4.

## Example

Kernel (`kernels/src/lib.rs`):

```rust
#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

/// # Safety
/// Buffers cover n elements; output is disjoint from inputs. The launch is 1D.
#[kernel]
pub unsafe fn vec_add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, out: WriteBuffer<f32>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        // SAFETY: the guard bounds each access; every output index has one writer.
        unsafe { out.store_unchecked(i, a.load_unchecked(i) + b.load_unchecked(i)) };
    }
}
```

Host (`host/src/main.rs`):

```rust
use metal_oxide::{Device, LaunchConfig};

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

    // SAFETY: disjoint three-element buffers, 1D launch, one writer per index.
    unsafe { kernels.vec_add(config, &a, &b, &mut out, 3)? };

    assert_eq!(out.as_slice(), &[11.0, 22.0, 33.0]);
    Ok(())
}
```

`cargo-metal` generates the bindings and supplies the build paths.
Complete Cargo setup: [examples/vec-add](https://github.com/rust-dd/metal-oxide/tree/main/examples/vec-add).

Precompiled hosts can commit the generated bindings, manifest and metallib, then
use `kernels::load_bytes(&device, include_str!("manifest.json"),
include_bytes!("kernels.metallib"))`. Pipelines are created on first use. These
hosts need stable Rust and the runtime; kernel compilation tools stay on the
build machine. See [generated bindings](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md#generated-rust-bindings).

Use `Device::all()?.into_iter().nth(index)` to select a device explicitly.
`recommended_max_working_set_size()` reports its advisory memory budget.
Cached runtime objects stay on one host thread; a [worker](https://github.com/rust-dd/metal-oxide/blob/main/crates/metal-oxide/tests/worker.rs)
can serve owned requests from multiple threads. Native Metal integration uses
`Device::from_native`, unsafe `Buffer::from_native`, and consuming `Buffer::into_native`.
See the [ownership contract](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md#native-resource-ownership).

Kernels can request runtime-sized block-local storage with
`DynamicThreadgroupBuffer<T>`. Generated bindings take an element count;
raw launches use `Argument::threadgroup::<T>(elements)`. Both paths validate
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

Verified on M4 Max, macOS 26.2, Xcode 26.6.

Math defaults to precise functions without implicit floating-point contraction.
Use `cargo metal build -p HOST --math-mode fast` to opt into fast functions,
reassociation, reciprocal approximations and contraction. Arithmetic keeps NaN
and infinity support; fast functions have narrower accuracy/domain guarantees
and zero signs are unspecified. The mode is recorded in the artifact and cache;
see [numerical compilation](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md#apple-compilation-and-artifact).

[Architecture](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md) · [Build, test, benchmarks](https://github.com/rust-dd/metal-oxide/blob/main/CONTRIBUTING.md) · [MIT](https://github.com/rust-dd/metal-oxide/blob/main/LICENSE)
