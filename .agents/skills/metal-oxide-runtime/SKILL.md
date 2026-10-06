---
name: metal-oxide-runtime
description: Use when changing metal-oxide Metal device, buffer, library, pipeline, argument, or kernel launch code and its hardware tests.
---

# Runtime changes

Read `AGENTS.md` and, when available locally, `docs/memory-model.md`.
The runtime is thread-confined and based on `objc2-metal`. It supports synchronous
launches and async ordered batches; Rust kernel compilation is separate.

- Keep the runtime buildable with stable Rust and independent of rustc internals.
- Keep classic `MTL*` and Metal 4 execution behind the same public API.
  Select the native path internally from OS/device capabilities; validate artifact
  requirements separately. Do not retry submitted work through another path.
  Metal 4 needs explicit residency and synchronization; classic hazard tracking
  does not apply. Verify each path independently on hardware before enabling it.
- Keep the public surface small. Add launch helpers, configuration variants, or
  wrapper layers only when a concrete current use case requires them. Prefer one
  clear path for each operation and keep implementation helpers private.
- Preserve initialized owned buffers and the sealed scalar boundary. Do not
  expose raw Metal handles or add Clone/Send/Sync without revisiting aliasing and
  resource ownership.
- Input arguments hold shared borrows; writes hold exclusive borrows. Submission
  lifetimes retain captured borrows until wait/await completes or the future is
  dropped. Pending buffer access state still synchronizes CPU slices after
  cancellation or `mem::forget`. Keep mutable slices on the host side only.
- Use `Device::launch` with `LaunchConfig<X, Y, Z>`: const block dimensions and
  a runtime `grid: Dim3`. Y and Z default to one; `for_elements` is a 1D helper.
  Keep the block shape in the type, without a mutable runtime copy. Use
  `DynamicLaunchConfig` for pipeline-derived or runtime-tuned block shapes.
  Both paths share validation. The grid counts blocks and each block maps to a
  Metal threadgroup via dispatchThreadgroups.
- Const host configuration does not specialize handwritten MSL. Keep shader
  shape assumptions in the unsafe caller contract until generated bindings and
  artifact metadata can enforce them for specialized Rust kernels.
- A launch checks queue-context ownership, binding slots, device axis limits, pipeline
  block volume, and dimension/count overflow; commits, waits, and checks terminal
  status. Errors must not release active resources.
- Artifact loading validates the ABI version, requirements, and library digest.
  Load the verified bytes rather than reopening their path. Check argument types,
  access modes, and any required block shape before encoding a launch.
- Keep launches unsafe until all kernel-specific ABI, bounds, access, and race
  preconditions are proved by the API. Document the actual caller obligations.
- Keep fast math explicit. Source compilation uses safe math and precise
  functions; a new numerical operation needs its own semantics.

Run the ordinary Rust checks in `AGENTS.md`. Run explicit hardware tests for
runtime changes and report the actual device/toolchain. The vec_add comparison
uses `0`, `1`, `255`, `256`, `257`, and `1_000_003` elements. Check complete-block
rounding and untouched padding with const blocks of 32 and 256 and a dynamic
execution-width block. Test block/thread indices in 1D/2D/3D for typed and dynamic
configurations when changing launch geometry. Use
compile-fail examples to verify public borrowing constraints when changing the
argument API.

`Device::submit` creates an ordered `Batch`; its closure encodes unsafe launches.
`Submission` implements `Future` and blocking `wait`. Native completion callbacks
retain buffers, pipelines, queues, and Metal 4 allocation/binding/residency state
independently of the future. Do not replace this with waiting in `Drop`.

Keep argument tables and constant buffers immutable after encoding. Metal 4
requires dispatch barriers within a batch and queue barriers between batches.
Do not reset or release a command allocator before GPU completion.

Run `bash scripts/test-gpu.sh` from the repository root to test both paths,
generated artifacts, kernel chains, and cancellation on a Metal 4 capable Mac.
