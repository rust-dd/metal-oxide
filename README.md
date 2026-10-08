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
Complete Cargo setup: [examples/vec-add](examples/vec-add).

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

Requires Rust 1.99.0, macOS 15+ on Apple Silicon, and Xcode with the Metal Toolchain.
From the repository root:

```sh
rustup toolchain install nightly-2026-10-04 --component rustc-dev --component rust-src --component llvm-tools
xcodebuild -downloadComponent MetalToolchain
cargo run -p cargo-metal -- doctor
cargo run -p cargo-metal -- run -p vec-add
```

Verified on M4 Max, macOS 26.2, Xcode 26.6.

[Architecture](CONCEPT.md) · [Build, test, benchmarks](CONTRIBUTING.md) · [MIT](LICENSE)
