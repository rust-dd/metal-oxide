# Changelog

## Unreleased

### Changed

- Saved artifacts change only with their kernels or recorded inputs. Another compiler
  build, machine or Metal toolchain that produces the same kernels leaves `metal/`
  unchanged. Manifests name the metal-oxide compiler version instead of a compiler
  binary hash, and recorded path packages no longer end in a space.
- Kernel items inside inline modules save their artifacts below the module names, such
  as `metal/src/lib.rs/tests/fill-precise/`, so same-named items in one file no longer
  overwrite each other's artifacts.

### Fixed

- Repeated integer casts retain proven slice bounds, including before threadgroup
  barriers. Changed indices and different casts keep their runtime checks.
- Without the Metal compiler, kernel builds use a current saved artifact instead of
  failing after the Rust build. Reused builds rebuild after a Metal compiler or SDK
  change, and hosts track the toolchain's version files, so Cargo expands their kernels
  again after an Xcode or Metal Toolchain update.
- `dependencies(...)` crates build with the features and `default-features` setting that
  the kernels' package declares, and renamed dependencies resolve to their package.
  They previously built with default features only.
- Metal libraries target the minimum macOS that their kernels need. Linking omitted the
  deployment target, so libraries built with the macOS 26 SDK used AIR 2.8 and macOS 15
  refused to load them, saved artifacts included. The next build rewrites their
  `kernels.metallib`.

### Upgrading

- Saved inputs use format 4. Build once with the kernel toolchain and commit the
  regenerated `metal/` directories; locked and toolchain-less builds report format 3
  artifacts as unsupported until then.
- `metal_oxide_build::KernelCrate` takes the module's `location` instead of its `file`.

## 0.1.0-alpha.1

Prereleases use dotted identifiers from `0.1.0-alpha.1` on. SemVer orders
`0.1.0-alpha.1` below `0.1.0-alpha0`, so `0.1.0-alpha0` is yanked once
`0.1.0-alpha.1` is published.

### Added

- Kernels support integer powers, checked division and remainder, comparisons and
  byte swaps, integer and float clamps and signs, and `f32::copysign` and `total_cmp`.
- `Device`, `Buffer`, `Pipeline`, `Module`, `Launch` and `Batch` implement `Debug`
  without reading buffer contents or waiting for GPU work.
- The examples index lists the command for each runnable example; crate metadata
  links to the hosted book.

### Changed

- `Error` and `KernelFailure` are non-exhaustive. `Result<T, E = Error>` also accepts
  a custom error type.
- BF16 buffers expose direct shared-storage CPU views, avoiding mirror copies.
- Generated-program tests compare both halves of `u64` results and detect missing
  writes and overwritten output padding. F16 graphs run independently of BF16;
  selecting a BF16 test on an unsupported device fails instead of passing.

### Fixed

- Generated MSL omits stores that nothing reads and keeps their checks. Kernels with
  such stores failed with "Compilation failed" in the macOS 15 Metal compiler.
- A missing or mismatched kernel compiler reports the matching `cargo-metal`
  install command.
- Kernel failures retain the original dispatch push index when earlier dispatches
  have empty grids.
- The book and README explain release-profile overflow checks and the scope of
  threadgroup barriers. CI verifies nextest downloads and enforces its version.

### Upgrading from 0.1.0-alpha0

1. Install the matching tools with
   `cargo install cargo-metal --version '=0.1.0-alpha.1' --locked`, then run
   `cargo metal setup`.
2. Change every metal-oxide dependency to `=0.1.0-alpha.1`.
3. Build once with the kernel toolchain and commit the regenerated `metal/`
   directories. Saved artifacts record the locked package versions, so locked and
   toolchain-less builds of your own crates report them as stale until then.
4. Add a wildcard arm to exhaustive `Error` matches, and `..` to `KernelFailure`
   patterns. Constructing `KernelFailure` outside the runtime is no longer supported.
   Compiler integrations matching `metal-oxide-ir` operation enums must handle
   their new comparison, byte-swap and copy-sign variants.

Artifacts retain ABI 4 and saved alpha0 dependencies remain loadable. The hardware
coverage and limitations recorded for alpha0 still apply; the two-week scheduled
fuzz observation period remains outstanding.

## 0.1.0-alpha0

### Recorded hardware coverage

| GPU | macOS | Xcode | Native APIs |
| --- | --- | --- | --- |
| M4 Max | 26.2 | 26.6 | classic Metal and Metal 4 |
| Apple Paravirtual device in GitHub-hosted M1 VMs | 15.7 and 26.6 | 16.4 and 26.6 | classic Metal |

The host supports Rust 1.89 and later. Development uses Rust 1.99; kernel authoring
uses nightly-2026-10-04 with rustc-dev, rust-src and llvm-tools. Saved artifacts
allow hosts and dependents to build without the kernel compiler. Artifacts use
ABI 4; incompatible contract or format changes after this release increment it.

### Known limitations

This is an early alpha. The API can change between releases.

Native M1–M3 GPUs and another Metal 4 GPU generation have not been verified;
Paravirtual execution does not establish native M1 coverage. Acquire/release
atomics require macOS 27/MSL 4.1 and still await hardware validation.
Automated rust-analyzer checks pass; editor rendering and Rust source display
in Xcode remain unverified. Two weeks of green scheduled fuzz runs have not yet
accumulated; that observation continues after this alpha.

Rust tensor primitives remain deferred after the MPP comparison. Unsupported
Rust operations, including recursion, trait-object calls, destructors, `f64`,
`i64` and 128-bit integers, are rejected explicitly. Fast math permits
reassociation and does not promise identical floating-point bits across compiler
versions. See [supported Rust](book/src/supported-rust.md),
[numerical modes](book/src/performance.md#numerical-modes) and
[configuration limitations](book/src/installation.md).

### Working features

- Ordinary Rust kernels in host functions, shared modules or separate `no_std`
  crates, compiled from rustc MIR through typed IR to Metal Shading Language.
- CUDA-style thread and block coordinates, static and dynamic block shapes,
  shared memory, SIMD groups, matrices and explicitly instantiated generic kernels.
- Typed buffers, checked launch contracts, GPU assertion reporting, async launches,
  operation chains and multiple launches in one command buffer.
- One runtime API for classic Metal and Metal 4, with device capability checks,
  native buffer ownership, validation, source inspection and GPU capture.
- Precise and fast numerical modes, explicit conversions, narrow integers,
  half types, supported atomics and small `core::simd` vectors.
- An authoring book, generated capability reference, packaged-installation checks,
  CPU/MSL/GPU differential tests and interleaved benchmarks.
