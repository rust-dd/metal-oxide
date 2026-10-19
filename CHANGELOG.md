# Changelog

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
