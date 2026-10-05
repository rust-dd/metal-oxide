# metal-oxide

Rust compute kernels for Metal on Apple Silicon.

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

Verified on Apple M4 Max, macOS 26.2, and Xcode 26.6.

[Contributing](CONTRIBUTING.md)

MIT licensed.
