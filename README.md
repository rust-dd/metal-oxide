# metal-oxide

Rust compute kernels for Metal on Apple Silicon.

## Compiler flow

```text
Rust kernel → rustc → MIR → metal-oxide IR → MSL → Metal compiler → .metallib
                                   └──────→ ABI + host bindings + manifest
Rust host → generated bindings → metal-oxide runtime → GPU
```

The compiler uses `nightly-2026-10-04`; the runtime uses stable Rust 1.99.0.
Launches follow CUDA's thread/block/grid model. Kernel launches are unsafe.
The runtime selects Metal 4 on macOS 26 with a compatible GPU and classic Metal
otherwise. `Device::submit` encodes ordered kernel batches that can be awaited.

## Build and run

Requires macOS 15+ on Apple Silicon and Xcode with the Metal Toolchain.

From the repository root:

```sh
rustup toolchain install nightly-2026-10-04 --component rustc-dev --component rust-src --component llvm-tools
xcodebuild -downloadComponent MetalToolchain
cargo install --path crates/cargo-metal --locked
cargo metal doctor
cargo metal run -p vec-add
cargo metal run -p reduction
cargo metal run -p matmul
cargo metal run -p pipeline
```

Inspect the generated MSL or run the GPU tests:

```sh
cargo metal inspect -p vec-add --emit msl
cargo metal test -p vec-add
bash scripts/test-gpu.sh
```

The host selects its kernel crate with `[package.metadata.metal]` and
`kernels = "../kernels/Cargo.toml"`. Builds write the library, manifest, and
bindings under `target/metal/<build-hash>/`.

Verified on Apple M4 Max, macOS 26.2, and Xcode 26.6.

[Contributing](CONTRIBUTING.md)

MIT licensed.
