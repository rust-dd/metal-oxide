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
- Keep future device builtins consistent with CUDA thread/block/grid semantics.
  Use `thread_idx()`, `block_idx()`, `block_dim()`, and `grid_dim()` with x/y/z
  fields; Metal threadgroups implement blocks. Scalar kernel parameters do not
  define launch dimensions automatically.
- Emit MSL through a tested structured control-flow subset. Unsupported MIR,
  assertions, recursion, layouts, and conversions must produce diagnostics.
  Never discard bounds/overflow assertions or silently change semantics.
- Specify buffer bindings and scalar representation in a versioned ABI. Do not
  copy the host memory layout of arbitrary Rust parameter aggregates.
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
-- --ignored --test-threads=1`. Tests explicitly select wrapping integers;
preserve enabled/always-on assertions as errors. When available locally, use
`docs/ir.md` for arithmetic and supported control-flow rules.

Include sources/dependencies, compiler/Rust/Metal/SDK versions, target
settings, and numerical options in artifact cache identity. Report the milestone
actually completed and keep planned APIs clearly marked.
