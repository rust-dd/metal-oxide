# metal-oxide

Rust compute kernels for Metal on Apple Silicon.

M2 compiles ordinary Rust functions from a separate `no_std` kernel crate to MSL
and runs the generated code through the Rust host runtime. `#[kernel]` marks an
entrypoint; rustc handles types, borrowing, and MIR generation.

## Compiler flow

```text
Rust kernel → rustc → MIR → metal-oxide IR → MSL → Metal compiler → GPU
Rust host   → metal-oxide runtime → buffers, pipeline, launch
```

The compiler uses `nightly-2026-10-04`; the runtime uses stable Rust 1.99.0.
Launches follow CUDA's thread/block/grid model. Kernel launch is currently unsafe.

## Run

Requires macOS 15+ on Apple Silicon. Compiler tests also require `clang++`.

Compile the Rust kernels and check their results on the GPU:

```sh
cd crates/metal-oxide-compiler
cargo test --features rustc-private --test gpu --locked --target-dir ../../target/compiler -- --ignored --test-threads=1
```

From the repository root, run the handwritten MSL reference:

```sh
cargo run --package vec-add --locked
```

The host also accepts `--source PATH` for a generated `kernels.metal` file.

## Current limits

- Scalars and buffer elements: `f32`, `u32`, `i32`; internal `bool`.
- Structured branches, single-exit loops, and concrete helper instances.
- Active MIR assertions, integer division, and float-to-integer casts are rejected.
  The GPU tests explicitly use `-C overflow-checks=off` for wrapping integers.
- MSL and an IR dump are emitted; CLI builds, manifests, and generated bindings
  are M3 work. `cargo metal` currently provides `doctor`.
- The runtime uses classic `MTL*`; Metal 4 support is planned for M6.

Verified on Apple M4 Max, macOS 26.2, and Xcode 26.6.

[Contributing](CONTRIBUTING.md)

MIT licensed.
