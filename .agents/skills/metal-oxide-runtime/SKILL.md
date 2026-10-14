---
name: metal-oxide-runtime
description: Use when changing metal-oxide Metal device, buffer, library, pipeline, argument, or kernel launch code and its hardware tests.
---

# Runtime changes

Read `AGENTS.md` and, when available locally, `docs/memory-model.md`.
The runtime is based on `objc2-metal`; its handles are `Send + Sync`. It supports
lazy launch operations that run when awaited or waited; Rust kernel compilation is
separate.

- Keep the runtime buildable with stable Rust and independent of rustc internals.
- Keep classic `MTL*` and Metal 4 execution behind the same public API.
  Select the native path internally from OS/device capabilities; validate artifact
  requirements separately. Do not retry submitted work through another path.
  Metal 4 needs explicit residency and synchronization; classic hazard tracking
  does not apply. Verify each path independently on hardware before enabling it.
- Keep the public surface small. Add launch helpers, configuration variants, or
  wrapper layers only when a concrete current use case requires them. Prefer one
  clear path for each operation and keep implementation helpers private.
- Preserve initialized CPU values and explicit GpuValue encoding/decoding.
  Built-in primitive scalars use direct shared-storage views after a private
  type-identity proof; custom codecs retain their CPU mirror. Layout equality
  alone does not authorize reinterpreting a custom Rust type.
  Atomic elements remain sealed to u32/i32. Validate codec size/alignment against
  the canonical layout; never reinterpret arbitrary Rust values as GPU bytes.
  `Buffer` is `Sync` because GPU writes need `&mut` and a codec mirror is decoded
  under its lock at most once per shared borrow; preserve that protocol.
- Native buffer import is an unsafe owning handoff of initialized, shared,
  hazard-tracked storage on the same physical device. External work must finish
  before import and external aliases must remain idle through CPU views and
  encoded/pending runtime accesses. Consuming export waits and encodes dirty
  codec values. Preserve identity storage, codec, cancellation and residency
  guarantees; dropping a buffer is not a synchronized handoff. Native device
  import creates independent queues; it does not adopt pending work.
- `Device` clones share its 4 command queues. A batch goes to the queue holding most of the
  in-flight accesses it conflicts with (writes before its reads, any access before its
  writes), else to the least busy queue. Completion arrives in commit order per queue, so
  independent work on other queues finishes independently (5.2 ms to 0.1 ms behind a long
  kernel on M4 Max). Classic tracked hazards do not order command buffers across queues:
  every classic command buffer sets a pooled event when its work finishes, and a batch waits
  for the events of conflicting work on other queues. Metal 4 signals a queue's timeline on
  demand after the work committed to it, and starts a batch with a queue barrier only when
  conflicting work runs on its own queue. `Batch` records dispatches and lives only
  inside the first poll or `wait` of its operation. Put `unsafe impl Send/Sync` only on small leaf types that own
  non-thread-safe native objects (buffer storage, Metal 4 command storage), with the
  protocol that makes them sound; never on `Device` or `Buffer` as a whole.
- Cross-thread state uses the `metal/sync.rs` primitives so the loom models in
  `metal/loom.rs` cover completion, buffer access and codec mirrors. Run them with
  `RUSTFLAGS="--cfg loom" cargo test -p metal-oxide --lib --release loom` after
  changing that state.
- Read arguments are shared borrows, owned buffers or `Arc`s; writes and read-write
  access need exclusive borrows or ownership. An operation and its future retain the
  arguments until wait/await completes or the future is
  dropped. Pending buffer access state still synchronizes CPU slices after
  cancellation or `mem::forget`. Keep mutable slices on the host side only.
- Launch kernels through `Launch` with `LaunchConfig<X, Y, Z>`: const block dimensions and
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
- A batch encodes at commit. Each dispatch goes to the first level after every level
  holding a dispatch that wrote a buffer it uses or read a buffer it writes; levels are
  encoded in order with a barrier between them, dispatches of one level in push order.
  Classic uses a serial encoder when every level holds one dispatch (barriers in a
  concurrent encoder cost 2-8% on such chains) and a concurrent encoder otherwise.
- Generated kernels bind a gate word after their parameters, their own failure word
  when fallible, and a table of argument element counts when they read buffer lengths.
  `Failures` owns these pooled words until completion. A level's dispatches share a gate
  that none of them writes, since all threads of a dispatch must agree on it: zero, the
  single failure word of an earlier level, or a fresh word that snapshot dispatches set
  after the previous barrier when several earlier words may be set.
- The handler reports the earliest-pushed fallible dispatch whose failure word is set as
  `Error::KernelFailed`; command errors take precedence. It then returns the cleared
  words to the pool.
- A batch that reads a buffer written by in-flight batches gates its generated dispatches
  on their words: before its first level, a snapshot covers every word of those batches,
  their snapshot gates included, so a failure propagates down a chain. An access entry
  holds a weak reference to its writer's `Failures`; the reader upgrades it at commit and
  keeps it until its own handler records the root failure, which releases the upstream.
  Without an own failure the handler reports the upstream root as
  `Error::DependencyFailed`. Finished writers do not propagate: their operations report. Generated MSL loads only through artifacts: source modules
  bind no words.
- Loading a `logging` artifact gives the device a `MTLLogState`; later command
  buffers carry it, and its handler prints messages to stderr.
- With `METAL_OXIDE_CAPTURE` set, the first `Device` of the process captures its native
  device (every queue of both backends) into that `.gputrace` until its context drops.
  Metal's capture proxies forward some messages, so objc2's encoding assertions, which
  reject forwarded methods in debug builds, stay disabled.
- Artifact loading validates the ABI version, requirements, and library digest.
  Load the verified bytes rather than reopening their path. Check argument types,
  access modes, any required block shape, the one-dimensional launch of a
  `one_dimensional` kernel, and the kernel's `requires` relations (argument lengths and
  integer values, checked `i128` arithmetic) before encoding a launch; a failed relation is
  `Error::Contract` with its source text.
- A safe `fn` kernel's generated method is safe: the device API keeps every racy or
  unchecked operation `unsafe`, and the launch checks types and `requires`. `Launch::new`
  and the methods of `unsafe fn` kernels stay unsafe; document the actual caller
  obligations.
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

An operation encodes nothing until its first poll or `wait`; then one `Batch` records
the whole operation (a `Launch`, or the operations a `sequence` closure pushes), encodes
it and commits. The futures (`Pending`, `TimedPending`, `Read`, `Write`) never block, and
`Completion` wakes every distinct waker. Native completion callbacks
retain buffers, pipelines, queues, and Metal 4 allocation/binding/residency state
independently of the future. Do not replace this with waiting in `Drop`.

Keep argument tables and constant buffers immutable after encoding. Metal 4
packs each dispatch's constants into aligned, immutable shared storage; classic
Metal copies values up to 4 KiB with setBytes and retains larger constant buffers.
Keep both size boundaries and cancellation/reuse covered when changing binding.
Dynamic threadgroup arguments have a separate 31-slot namespace; check rounded
nonzero byte lengths plus static pipeline storage before encoding, including empty grids.
Clear stale threadgroup slots when a reused encoder changes kernels.
Metal 4 requires encoder barriers between dependent dispatches and a queue barrier before a
batch that follows work on its queue.
Do not reset or release a command allocator before GPU completion.
The context pools the command buffer, allocator, residency set and binding storage
of up to 64 committed batches, like a classic queue's in-flight command buffers. A fresh
allocator costs a kernel call on its first encoder, so with every pooled batch running
the next batch waits for the oldest instead (back-to-back launches: 28 to 10 µs each on
M4 Max). Reuse requires completed feedback and an empty residency set; reset the
allocator only then. Retain constant allocations only up to 4 KiB each, rewrite active
bindings and clear unused slots. Each dispatch owns distinct immutable binding storage
through completion. Failed encoding must remain safe without waiting in Drop.

Run `bash scripts/test-gpu.sh` from the repository root to test both paths,
generated artifacts, kernel chains, and cancellation on a Metal 4 capable Mac.
