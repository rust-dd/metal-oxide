# metal-oxide

Rust compute kernels for Metal on Apple Silicon.

Kernels are ordinary Rust functions in a separate `no_std` crate. A Rust host
application loads the compiled kernels and launches them through `metal-oxide`.

## Architecture

```text
       [Rust kernel]
             |
        [rustc: MIR]
             |
       [importer: IR]
             |
         [codegen] ----> ABI + bindings
             |                 |
            MSL                |
             |                 |
      [Apple compiler]         |
             |                 |
         .metallib             |
             +-----------------+
             |
   [Rust host + runtime]
             |
 [Classic Metal / Metal 4]
```

`rustc` handles parsing, macros, type checking, borrow checking, and MIR generation.
The MIR importer resolves concrete function instances and converts them to our
typed IR. The IR records control flow, address spaces, and buffer access. Codegen
emits Metal Shading Language (MSL), which Apple's compiler turns into a shader
library. `#[kernel]` marks entrypoints; it does not translate function bodies.

| Crate | Role |
| --- | --- |
| `metal-oxide-device` | `no_std` buffer handles, thread indices, shared memory, atomics, and SIMD-group operations |
| `metal-oxide-macros` | `#[kernel]` entrypoint and block-shape markers |
| `metal-oxide-compiler` | `rustc_driver` integration, function collection, and MIR import |
| `metal-oxide-ir` | Typed intermediate representation and validation |
| `metal-oxide-codegen` | MSL, kernel ABI, and Rust host binding generation |
| `metal-oxide-artifact` | Versioned ABI, manifest format, and artifact verification |
| `metal-oxide` | Typed buffers, library loading, compute pipelines, and execution |
| `cargo-metal` | Kernel build, Apple compiler invocation, artifact cache, and host build |

The ABI defines value layouts, buffer strides and slots, access modes, and
required block shapes. Generated record types encode fields and padding explicitly;
their Rust memory layout is independent of the GPU layout. The MSL signature
and generated Rust bindings use the same ABI. The runtime checks this metadata
and the library hash when loading an artifact, then validates arguments before
encoding a launch.

The compiler uses `nightly-2026-10-04`; the runtime uses stable Rust 1.99.0.
The runtime has no dependency on the compiler or `rustc_private`. `cargo metal`
runs the compiler separately and builds the host with the generated bindings.

See [CONCEPT.md](CONCEPT.md) for the `vec_add` compiler, ABI, and runtime walkthrough.

## Runtime

A `Device` owns a GPU and command queue. A `Module` loads a shader library;
a `Pipeline` selects one kernel entrypoint. `Buffer<T>` owns GPU memory, and
`Argument` binds a buffer or an encoded value to a parameter slot. Generated
bindings construct these arguments from typed Rust parameters.

`LaunchConfig<X, Y, Z>` sets threads per block through const generics. Its `grid`
counts blocks at runtime. Each block maps to a Metal threadgroup;
`thread_idx()`, `block_idx()`, `block_dim()`, and `grid_dim()` follow CUDA's model.

`Device::launch` runs a kernel and waits for completion. `Device::submit` encodes
ordered kernels in a `Batch` and returns a `Submission` that supports `.await`
and `.wait()`. Completion callbacks retain GPU resources until execution ends.
CPU buffer access waits for pending GPU work even if the submission is dropped.
Kernel launches are unsafe: callers must satisfy the kernel's bounds and race
requirements.

The runtime selects Metal 4 on macOS 26 with a compatible GPU and classic Metal
otherwise. Both backends use the same public API.

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
cargo metal run -p particle-update
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
