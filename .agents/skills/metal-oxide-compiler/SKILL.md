---
name: metal-oxide-compiler
description: Use when implementing metal-oxide rustc integration, device APIs, MIR import, typed IR, MSL emission, or kernel artifact ABI.
---

# Compiler changes

Read `AGENTS.md`. When available locally, also read `docs/architecture.md`,
`docs/roadmap.md`, and relevant sections of `docs/supported-rust.md`.
Implement the next acceptance criterion; create crates when their code is needed.

- Use typed rustc APIs through a pinned `rustc_driver` integration. A kernel
  proc macro marks entries; it does not translate Rust bodies. Do not parse
  textual MIR or substitute a Rust source parser for type/borrow checking.
- Choose a dated nightly after a real compatibility probe. Define device
  pointer/usize widths, layouts, cfgs, and matching core metadata/MIR before
  accepting separate no_std device crates. Host layouts are not device layouts.
- Collect concrete kernel entries and reachable concrete function instances.
  Generic buffer handles and helpers need monomorphization from the start.
- Carry const block dimensions from typed launch configurations into
  shape-dependent kernel variants. Record the shape in artifact metadata and
  cache keys; generated launch functions must enforce the matching configuration type
  or validate a dynamic shape. Host const generics alone do not specialize MSL.
- Keep IR/codegen/artifact contracts independent of rustc_private. Track types,
  source locations, address spaces, and resource access explicitly in IR.
- Keep device builtins consistent with CUDA thread/block/grid semantics.
  Use `thread_idx()`, `block_idx()`, `block_dim()`, and `grid_dim()` with x/y/z
  fields; Metal threadgroups implement blocks. Scalar kernel parameters do not
  define launch dimensions automatically. `thread_index()` returns a zero-sized witness;
  `ThreadIndex::get` and `DisjointBuffer::get_mut` read `ThreadIndex`
  (`thread_position_in_grid.x`), which proofs treat as one value however often it is read.
  A kernel that reads it, directly or through a helper, records `one_dimensional` in its
  ABI, and the runtime rejects other launches.
- A safe `fn` kernel records `safe` in its ABI and gets a safe launch function.
  `#[kernel(requires(...))]` relations are parsed from their source text (lengths of buffer
  parameters, integer parameters, literals, `+ - *`, one comparison) into the kernel's IR and
  ABI; proofs assume those over plain atoms at entry, and an assumed equality makes its two
  sides interchangeable.
- Precede emitted statements with `#line` directives for their Rust source (short form within
  one file) and reset to `kernels.metal` after each body; directives never change AIR.
- Emit MSL through a tested structured control-flow subset. Unsupported MIR,
  recursion, layouts, and conversions must produce diagnostics that describe the
  construct, never print MIR. Never remove a bounds/overflow check without a sound proof
  or silently change semantics.
- Report rejections in the kernel crate's source: the collector and importer move a
  rejection in another crate's code to the kernel call that reached it with a
  `REACHED` note; IR errors carry the calls the prover walked, and the driver maps their
  locations back to spans. Another crate's macro expansion locates at its kernel call.
- Unproved assertions and literal panics become runtime checks: the failing thread
  stores `check + 2` in its dispatch's failure word and returns; callers of
  fallible helpers return after them. A check before a barrier or SIMD collective
  must be proved or uniform. Generated kernels take a gate word in the buffer slot
  after their parameters, then the failure word when fallible, then a `ulong` table of
  parameter lengths when they read buffer lengths; user kernels keep 28 slots.
- A kernel tests its gate once, at the latest point that dominates every effect and
  outside loops. On M3/M4 GPUs a gate at kernel entry, extra gate sites or a richer
  failure path lower the occupancy of the whole kernel; keep failure recording to one
  atomic store.
- Proofs treat launch coordinates as related values: launch validation keeps
  `grid_dim * block_dim` within `u32` and a kernel's required block exact, so the
  global index needs no overflow check. Summaries are keyed by the required block.
- `log!` lowers to MSL `os_log` with typed printf specifiers; artifacts that use it
  require the `logging` feature, MSL 3.2 and `-fmetal-enable-logging`.
- Enums, `Option` and `Result` included, are IR aggregates: a tag (the narrowest
  unsigned integer that numbers the variants), then payload slots that variants share by
  exact type, each variant taking the first free slot of its field's type. Only default
  discriminants are supported, with `core::cmp::Ordering` represented as its signed byte;
  enums never cross the ABI. A variant field projects
  onto its slot, a discriminant reads the tag, and an exhaustive `match` lets its last case
  take the unreachable default.
- `&str` and `&dyn Trait` values have no device representation. A message argument
  specializes its callee per literal, so `expect` reaches core's `expect_failed` with a
  known message; core's `unwrap_failed`, `expect_failed` and `result::unwrap_failed` end
  the caller's block with a check. A `#[track_caller]` function is instantiated per
  caller, and its checks report the caller's `fn_span`, as Rust does.
- A closure is a record of its captures; captured references keep their address spaces
  like any aggregate's. `Fn`, `FnMut` and `FnOnce` calls use the `rust-call` ABI: a call
  spreads its argument tuple into a closure body's parameters, while rustc's call shims
  (closure-once, and fn items called through the `Fn` traits) take the tuple.
- Zero-sized values (`PhantomData`, field-less structs, fn items, closures without
  captures) and references to them lower to unit: records keep them as fields without
  storage, and their assignments emit nothing.
- Uninhabited types lower to `Never`; a variant with an uninhabited field has no payload,
  so `?` on `Result` builds its `Result<Infallible, E>` residual.
- Core's iterators run core's MIR. Only `Iterator::next` on an integer `Range` and on
  `ChunksExact` expands inline, because proofs keep loop bounds and chunk lengths only
  when `next` is inlined; a `&mut` borrow of such an iterator is erased only when every
  use advances it, and a `match` on the result branches straight to its arms.
- Each instance's MIR is pruned before import, as codegen would: `mono_reachable`
  removes branches on generic constants (`T::IS_ZST`), runtime-check switches and
  `check_language_ub` take their configured arm, `cold_path` hints are dropped, and blocks
  that can only reach `unreachable` become unreachable, so switches drop those arms. Core's
  unsafe-precondition checks are off (`-Zub-checks=no`, as in release builds).
- Raw pointers are IR pointers whose spaces are inferred like references'. A transparent
  struct around a pattern type (`NonNull<T>`, `NonZero<T>`) is the value it holds.
  Pointer casts keep the pointee and address space; reinterpreting casts are rejected.
  `isize` is an internal scalar. A transmute into an `Option`-like enum stored in an
  integer niche (`NonZero::new`) computes the tag from the niche value.
- A callee whose return values hold no reference of their own, such as `None` from `?`,
  takes its return type from the caller's destination.
- Proofs resolve the referent of an exclusive `&mut` parameter (only read and written
  through, never copied, passed or mutably reborrowed) like a shared referent.
- Specify buffer bindings and scalar representation in a versioned ABI. Do not
  copy the host memory layout of arbitrary Rust parameter aggregates.
- Records, tuples and arrays inside kernels may hold references. Each reference keeps the
  address space of the value it was built from, inferred like reference locals; an IR type
  lists its spaces in walk order (a reference before its target, fields in order, an
  array's element once), and helpers are instantiated per argument type. Only owned values
  cross the ABI; keep their internal shape separate from the host ABI. Cross-crate helpers require matching encoded MIR. Preserve
  concrete type/const instances instead of emitting their generic definitions.
- Validate cooperative participation transitively through helpers. Varying
  branches, early returns, and loops must not bypass required barriers or SIMD
  collectives. Threadgroup allocations have distinct storage per callsite.
- The kernel macros generate one launch function per entrypoint from its signature,
  generic over the borrowed, owned and `Arc` buffer forms, returning its `Launch`;
  they check it against the built ABI and enforce the const block shape. Kernels in a
  host crate build from a copy of their file in which everything outside the item is
  blanked, so spans keep their lines and columns.
- Keep MSL supported; direct AIR is deferred until measurements justify it.
- Within-thread `core::simd` uses scalar lanes for 2–4 elements of f32 or 8/16/32-bit
  integers. Keep core's checks and lower supported vector intrinsics explicitly. Vector
  types never cross the ABI; wider vectors and memory gather/scatter are rejected.

Run compiler tests and Clippy from `crates/metal-oxide-compiler` with
`--features rustc-private --locked --target-dir ../../target/compiler`. Its
dated toolchain is separate from the stable workspace toolchain. The frontend
tests build matching device `core`/`compiler_builtins` metadata and compile the
marker macro natively; preserve that target separation.

Use UI diagnostics, IR/MSL checks, ABI validation, and GPU comparisons as the
implementation reaches those boundaries. Generated-kernel acceptance tests must execute
MSL generated from Rust kernel source. A handwritten replacement,
synthetic IR example, or MIR dump does not satisfy that criterion.

Run generated-kernel hardware tests from the compiler directory with
`cargo test --features rustc-private --test gpu --locked --target-dir ../../target/compiler
-- --ignored --test-threads=1`. Kernel builds use the release profile, so overflow
checks follow it; bounds, division and explicit panics are always checked. Tests of
proofs assert which checks remain in `checks.json`. When available locally, use
`docs/ir.md` for arithmetic and supported control-flow rules.

`bash scripts/test-gpu.sh` from the root verifies both native runtime APIs and
offline artifacts. The `pipeline` example covers struct helpers, const
specializations, async launches, and ordered kernel chains.

Include sources/dependencies, compiler/Rust/Metal/SDK versions, target
settings, and numerical options in artifact cache identity. `metal-oxide-build`
drives kernel builds for the kernel macros; each build saves its artifact in the
crate's `metal/` directory with its inputs in `inputs.json`: the item's source digest,
locked packages and path packages' files. Fetched packages, builds without the toolchain
and `METAL_OXIDE_ARTIFACTS=locked` use the saved artifact while those are unchanged. Report the milestone
actually completed and keep planned APIs clearly marked.
