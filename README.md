# metal-oxide

A Metal-first Rust compute compiler for macOS on Apple Silicon, with explicit
GPU threads, a rustc frontend, an MSL backend, and a separate Rust runtime.

The project is at **M1: Rust frontend**. A separate `no_std` Rust kernel crate
passes rustc type and borrow checking on a Metal device target. The compiler
collects typed MIR and concrete device/helper instances. The host runtime runs
the handwritten Metal `vec_add`; Rust-to-MSL generation is the next milestone.

## Compiler flow

Kernels are ordinary Rust functions in a separate `no_std` crate, using explicit
GPU threads and device buffer handles. `#[kernel]` marks an entrypoint; it does
not translate the function body. Rust syntax, types, and borrowing are handled
by `rustc`.

```text
Rust kernel crate + metal-oxide-device
    -> rustc parsing, macro expansion, type checking, borrow checking
    -> kernel entrypoints and reachable concrete function instances
    -> typed MIR
    -> metal-oxide IR and GPU validation
    -> Metal Shading Language + kernel metadata
    -> Apple Metal compiler
    -> .metallib + manifest + generated Rust bindings
    -> Rust host -> metal-oxide runtime -> Metal GPU
```

M0 provides the runtime and handwritten MSL execution. M1 adds the Rust frontend
and device target; M2 adds IR and MSL generation. Versioned artifacts and generated
host bindings follow in M3.

## Rust kernel frontend

The [Rust vec_add](examples/vec-add/kernels/src/lib.rs) uses the device API:

```rust
use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[kernel]
pub unsafe fn vec_add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, out: WriteBuffer<f32>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        unsafe { out.store_unchecked(i, a.load_unchecked(i) + b.load_unchecked(i)) };
    }
}
```

The caller guarantees buffer bounds, non-overlapping output, and one writer per
index in a one-dimensional launch. Device functions `thread_idx()`, `block_idx()`,
`block_dim()`, and `grid_dim()` return x/y/z coordinates with CUDA semantics.
They are compiler builtins and panic if called on the CPU.

The compiler toolchain is pinned separately in
[rust-toolchain.toml](crates/metal-oxide-compiler/rust-toolchain.toml), after checking
the actual device and kernel crates with `nightly-2026-10-04`. To run that check:

```sh
cd crates/metal-oxide-compiler
cargo test --features rustc-private --locked --target-dir ../../target/compiler
cargo clippy --features rustc-private --all-targets --locked --target-dir ../../target/compiler -- -D warnings
```

These tests build matching `core` and `compiler_builtins` metadata with device
MIR, compile the marker macro for the host, and analyze the separate device and
kernel crates. They check concrete type/const generic helpers, Rust diagnostics,
kernel signatures, unsupported calls, and retained overflow assertions. The
compiler reports entries, parameter types/access, function instances, and typed
MIR locals; it does not emit MSL or execute Rust kernels yet.

The device target uses 64-bit pointers and `usize`, little-endian layout, and
`target_env = "metal"`. Its pointer-handle layout is internal compiler metadata;
host launch bindings will use the explicit artifact ABI. The runtime stays on
stable Rust and does not link the compiler.

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
- A pinned rustc frontend, separate device/kernel crates, typed MIR collection,
  concrete helper/device instances, and frontend diagnostics.

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
