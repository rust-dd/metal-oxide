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
  cache keys; generated bindings must enforce the matching configuration type
  or validate a dynamic shape. Host const generics alone do not specialize MSL.
- Keep IR/codegen/artifact contracts independent of rustc_private. Track types,
  source locations, address spaces, and resource access explicitly in IR.
- Keep device builtins consistent with CUDA thread/block/grid semantics.
  Use `thread_idx()`, `block_idx()`, `block_dim()`, and `grid_dim()` with x/y/z
  fields; Metal threadgroups implement blocks. Scalar kernel parameters do not
  define launch dimensions automatically.
- Emit MSL through a tested structured control-flow subset. Unsupported MIR,
  recursion, layouts, and conversions must produce diagnostics. Never remove a
  bounds/overflow check without a sound proof or silently change semantics.
- Unproved assertions and literal panics become runtime checks: the failing thread
  stores `check + 2` in the command buffer's failure word and returns; callers of
  fallible helpers return after them. A check before a barrier or SIMD collective
  must be proved or uniform. Generated kernels take a gate word in the buffer slot
  after their parameters and, when fallible, the failure word in the next one.
- A kernel tests its gate once, at the latest point that dominates every effect and
  outside loops. On M3/M4 GPUs a gate at kernel entry, extra gate sites or a richer
  failure path lower the occupancy of the whole kernel; keep failure recording to one
  atomic store.
- Proofs treat launch coordinates as related values: launch validation keeps
  `grid_dim * block_dim` within `u32` and a kernel's required block exact, so the
  global index needs no overflow check. Summaries are keyed by the required block.
- `log!` lowers to MSL `os_log` with typed printf specifiers; artifacts that use it
  require the `logging` feature, MSL 3.2 and `-fmetal-enable-logging`.
- `Iterator::next` on `Range<integer>` expands inline into a bounds branch. Its
  `Option` result is an internal (`is_some`, `value`) record that kernels may read
  but not construct; `&mut` borrows are erased only as its argument. A `match` on
  the result branches straight to its arms. Other iterators, enum construction and
  references are rejected.
- Specify buffer bindings and scalar representation in a versioned ABI. Do not
  copy the host memory layout of arbitrary Rust parameter aggregates.
- Device records contain owned scalar fields; keep their internal shape separate
  from the host ABI. Cross-crate helpers require matching encoded MIR. Preserve
  concrete type/const instances instead of emitting their generic definitions.
- Validate cooperative participation transitively through helpers. Varying
  branches, early returns, and loops must not bypass required barriers or SIMD
  collectives. Threadgroup allocations have distinct storage per callsite.
- Emit one binding per entrypoint that returns its `Launch` operation, generic over
  the borrowed, owned and `Arc` buffer forms, and enforce the const block shape and
  parameter types.
- Keep MSL supported; direct AIR is deferred until measurements justify it.

Run compiler tests and Clippy from `crates/metal-oxide-compiler` with
`--features rustc-private --locked --target-dir ../../target/compiler`. Its
dated toolchain is separate from the stable workspace toolchain. The frontend
tests build matching device `core`/`compiler_builtins` metadata and compile the
marker macro natively; preserve that target separation.

Use UI diagnostics, IR/MSL checks, ABI validation, and GPU comparisons as the
implementation reaches those boundaries. Generated-kernel acceptance tests must execute
MSL generated from the separate Rust kernel crate. A handwritten replacement,
synthetic IR example, or MIR dump does not satisfy that criterion.

Run generated-kernel hardware tests from the compiler directory with
`cargo test --features rustc-private --test gpu --locked --target-dir ../../target/compiler
-- --ignored --test-threads=1`. Kernel builds use the release profile, so overflow
checks follow it; bounds, division and explicit panics are always checked. Tests of
proofs assert which checks remain in `checks.json`. When available locally, use
`docs/ir.md` for arithmetic and supported control-flow rules.

`bash scripts/test-gpu.sh` from the root verifies both native runtime APIs and
offline artifacts. The `pipeline` example covers external struct helpers,
const specializations, async binding calls, and ordered kernel chains.

Include sources/dependencies, compiler/Rust/Metal/SDK versions, target
settings, and numerical options in artifact cache identity. `metal-oxide-build`
drives kernel builds for `cargo-metal` and host build scripts; a committed artifact
records its source files and locked packages in `inputs.json` and is rebuilt only
when they change. Report the milestone
actually completed and keep planned APIs clearly marked.
