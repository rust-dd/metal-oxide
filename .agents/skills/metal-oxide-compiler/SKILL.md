---
name: metal-oxide-compiler
description: Use when implementing metal-oxide rustc integration, device APIs, MIR import, typed IR, MSL emission, or kernel artifact ABI.
---

# Compiler changes

Read `AGENTS.md`, `docs/architecture.md`, `docs/roadmap.md`, and the relevant
sections of `docs/supported-rust.md`. Implement the next acceptance criterion;
create crates when their code is needed. The initial checkout implements M0 only.

- Use typed rustc APIs through a pinned `rustc_driver` integration. A kernel
  proc macro marks entries; it does not translate Rust bodies. Do not parse
  textual MIR or substitute a Rust source parser for type/borrow checking.
- Choose a dated nightly after a real compatibility probe. Define device
  pointer/usize widths, layouts, cfgs, and matching core metadata/MIR before
  accepting separate no_std device crates. Host layouts are not device layouts.
- Collect concrete kernel entries and reachable concrete function instances.
  Generic buffer handles and helpers need monomorphization from the start.
- Keep IR/codegen/artifact contracts independent of rustc_private. Track types,
  source locations, address spaces, and resource access explicitly in IR.
- Emit MSL through a tested structured control-flow subset. Unsupported MIR,
  assertions, recursion, layouts, and conversions must produce diagnostics.
  Never discard bounds/overflow assertions or silently change semantics.
- Specify buffer bindings and scalar representation in a versioned ABI. Do not
  copy the host memory layout of arbitrary Rust parameter aggregates.
- Keep MSL supported; direct AIR is deferred until measurements justify it.

Use UI diagnostics, IR/MSL checks, ABI validation, and GPU comparisons as the
implementation reaches those boundaries. The M2 acceptance test must execute
MSL generated from the separate Rust kernel crate. A handwritten replacement,
synthetic IR example, or MIR dump does not satisfy that criterion.

At M3, include sources/dependencies, compiler/Rust/Metal/SDK versions, target
settings, and numerical options in artifact cache identity. Report the milestone
actually completed and keep planned APIs clearly marked.
