---
name: metal-oxide-runtime
description: Use when changing metal-oxide Metal device, buffer, library, pipeline, argument, or kernel launch code and its hardware tests.
---

# Runtime changes

Read `AGENTS.md` and `docs/memory-model.md`. The current runtime is synchronous,
thread-confined, and based on `objc2-metal`; Rust kernel compilation is separate.

- Keep the runtime buildable with stable Rust and independent of rustc internals.
- Preserve initialized owned buffers and the sealed scalar boundary. Do not
  expose raw Metal handles or add Clone/Send/Sync without revisiting aliasing and
  resource ownership.
- Input arguments hold shared borrows; writes hold exclusive borrows until GPU
  completion. Keep the full-buffer mutable slice on the host side only.
- Use `Device::launch` with `LaunchConfig { grid, block }` and `Dim3`. The grid
  counts blocks, the block counts threads per block, and each block maps to a
  Metal threadgroup via dispatchThreadgroups. Keep these CUDA-style semantics.
- A launch checks device ownership, binding slots, device axis limits, pipeline
  block volume, and dimension/count overflow; commits, waits, and checks terminal
  status. Errors must not release active resources.
- Keep launches unsafe until all kernel-specific ABI, bounds, access, and race
  preconditions are proved by the API. Document the actual caller obligations.
- Keep fast math explicit. Source compilation uses safe math and precise
  functions; a new numerical operation needs its own semantics.

Run the ordinary Rust checks in `AGENTS.md`. Run explicit hardware tests for
runtime changes and report the actual device/toolchain. The vec_add comparison
uses `0`, `1`, `255`, `256`, `257`, and `1_000_003` elements. Check complete-block
rounding and untouched padding with execution-width and explicit block sizes.
Test block/thread indices in 1D/2D/3D when changing launch geometry. Use
compile-fail examples to verify public borrowing constraints when changing the
argument API.

An async API needs a submission that retains resources/access state through GPU
completion even after future cancellation. Treat it as the M6 design task;
`Drop` waiting by itself does not establish the needed ownership.
