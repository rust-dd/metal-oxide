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
- CUDA-style 1D/2D/3D launches with const-generic block shapes, runtime grids,
  checked device/pipeline limits, and synchronous completion/error handling.
  Runtime-selected blocks are also supported. Empty grids are no-ops.
- `cargo metal doctor`; explicit hardware tests and portable launch checks.

Kernel launch is `unsafe`: the caller supplies the kernel ABI, memory bounds,
access modes, and race-freedom contract. See [the memory model](docs/memory-model.md).
The runtime has no dependency on rustc internals.

## Launch geometry

`grid` counts blocks, as in CUDA. Block dimensions are const generics on
`LaunchConfig<X, Y, Z>`; Y and Z default to one. Each Metal threadgroup implements
one block. The runtime uses `dispatchThreadgroups`.

```rust
use metal_oxide::{Dim3, LaunchConfig};

let config = LaunchConfig::<256>::new(Dim3::x(4));
let tiled = LaunchConfig::<16, 16>::new(Dim3::xy(2, 3));
```

`config` launches 4 blocks of 256 threads, or 1024 threads total. `tiled` launches
a 2-by-3 grid of 16-by-16 blocks. `Device::launch`
takes the pipeline, this configuration, and the `Argument` list. Block dimensions
are checked against device axis limits and the pipeline's total block limit.
The compile-time shape is available as `LaunchConfig::<256>::BLOCK`.

For a one-dimensional element range, `LaunchConfig::<256>::for_elements(n)?`
rounds up to complete blocks. For example, 1000 elements with a block size of 256
launch 1024 threads; the kernel's `i < n` condition guards the extra 24 threads.
The example uses a const block size of 256 and checks it against pipeline limits.

Use `DynamicLaunchConfig::new(grid, block)` or
`DynamicLaunchConfig::for_elements(n, block_size)?` for runtime-selected block
sizes, such as values from `Pipeline::thread_execution_width()`. `Device::launch`
accepts both configuration types and applies the same validation.

Const block dimensions specialize the Rust configuration type. They do not
specialize handwritten MSL. The planned Rust compiler will carry block constants
into kernel variants and generate bindings that enforce the matching shape.

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
- [NVIDIA CCCL: compile-time launch configuration](https://developer.nvidia.com/blog/cccl-runtime-a-modern-c-runtime-for-cuda/).

MIT licensed.
