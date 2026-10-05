# metal-oxide

A Metal-first Rust compute compiler for macOS on Apple Silicon, with explicit
GPU threads, a rustc frontend, an MSL backend, and a separate Rust runtime.

The project is at **M0: runtime foundation**. A Rust host runs a handwritten
Metal `vec_add` and checks its output. Rust kernel compilation starts at M1;
there is no Rust-to-MSL compiler in this checkout yet.

## Run the runtime example

Requirements: macOS 15 or newer on Apple Silicon and Rust 1.99.0. Xcode and its
Metal Toolchain component are needed for offline `.metallib` compilation.

```sh
cargo metal doctor
cargo run --package vec-add --locked
```

The example compiles [vec_add.metal](examples/vec-add/kernels/vec_add.metal) through
the Metal runtime. It checks `0`, `1`, `255`, `256`, `257`, and `1_000_003` elements
against a CPU reference. Source compilation can work without the separate Xcode
Metal Toolchain; `doctor` still returns a failure until the offline compiler is
available.

To compile and load the same kernel as a `.metallib`:

```sh
xcodebuild -downloadComponent MetalToolchain
mkdir -p target/metal
xcrun -sdk macosx metal -fno-fast-math -c examples/vec-add/kernels/vec_add.metal -o target/metal/vec_add.ir
xcrun -sdk macosx metal target/metal/vec_add.ir -o target/metal/vec_add.metallib
cargo run --package vec-add --locked -- --metallib target/metal/vec_add.metallib
```

The precompiled example expects the exact reference kernel ABI. Artifact
metadata validation and generated bindings belong to M3.

## What works

- Owned, initialized shared buffers of `f32`, `u32`, and `i32`.
- Read and exclusive write argument borrows, plus explicit four-byte scalars.
- Source and `.metallib` library loading; compute pipeline creation.
- CUDA-style grid/block launches in 1D/2D/3D, checked device/pipeline limits,
  and synchronous completion/error handling. Empty grids are no-ops.
- `cargo metal doctor`; explicit hardware tests and portable launch checks.

Kernel launch is `unsafe`: the caller supplies the kernel ABI, memory bounds,
access modes, and race-freedom contract. See [the memory model](docs/memory-model.md).
The runtime has no dependency on rustc internals.

## Launch geometry

`grid` counts blocks and `block` counts threads per block, as in CUDA. Each
Metal threadgroup implements one block. The runtime uses `dispatchThreadgroups`.

```rust
use metal_oxide::{Dim3, LaunchConfig};

let config = LaunchConfig {
    grid: Dim3::new(4, 1, 1),
    block: Dim3::new(256, 1, 1),
};
```

This launches 4 blocks of 256 threads, or 1024 threads total. `Device::launch`
takes the pipeline, this configuration, and the `Argument` list. Block dimensions
are checked against device axis limits and the pipeline's total block limit.

For a one-dimensional element range, `LaunchConfig::for_elements(n, block_size)?`
rounds up to complete blocks. For example, 1000 elements with a block size of 256
launch 1024 threads; the kernel's `i < n` condition guards the extra 24 threads.
The example chooses its block size from the pipeline's execution width.

`Dim3::x(count)` and `Dim3::xy(width, height)` are shortcuts for one- and
two-dimensional shapes. General 2D/3D launches need kernels that use matching
coordinates; the reference `vec_add` uses a one-dimensional grid and block.

## Checks

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo test --package metal-oxide --test gpu --locked -- --ignored --test-threads=1
```

The ordinary test command does not execute hardware tests. The last command
requires a real Apple Silicon Metal device.

M0 source execution and 1D/2D/3D block/thread indexing were verified on Apple
M4 Max, macOS 26.2, Xcode 26.6, and Rust 1.99.0. Broader device and toolchain
compatibility is not yet established.

See [the roadmap](docs/roadmap.md), [architecture](docs/architecture.md),
[supported Rust](docs/supported-rust.md), and [contributing guide](CONTRIBUTING.md).
Project development skills live in [.agents/skills](.agents/skills).

## References

- [Apple: precompiling shader libraries](https://developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files).
- [objc2-metal: binding safety requirements](https://docs.rs/objc2-metal/0.3.2/objc2_metal/).
- [cuda-oxide](https://github.com/NVIDIA/cuda-rust), a reference for Rust GPU compilation.

MIT licensed.
