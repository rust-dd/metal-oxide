# metal-oxide

Build a Metal-first Rust compute compiler for macOS on Apple Silicon.
Implement one milestone at a time. Keep planned APIs clearly separate from
working features. Add crates and modules when they have an implementation.

## Git and writing

- Use Conventional Commit subjects: `feat: ...`, `fix: ...`, `chore: ...`,
  `docs: ...`, `test: ...`, `refactor: ...`, `perf: ...`, or `ci: ...`.
- Do not add co-author trailers, AI attribution, session IDs, tool transcripts,
  or generated-by footers to commits, PRs, code, or project documentation.
- Write concise English code and documentation. Describe actual behavior.
- Comments explain safety contracts or non-obvious constraints. Do not add
  banners, separator lines, edit narration, commented-out code, or prose that
  repeats the next statement. Keep necessary `SAFETY` comments and rustdoc.
- Keep editor settings local. Commit the compiler's `rustc_private`
  rust-analyzer metadata in its `Cargo.toml`.
- Keep `docs/` local and ignored. Read its planning notes when present; do not
  commit them.
- Use names and small functions to express intent. Rustdoc documents behavior
  and caller obligations; skip trivial explanations and implementation narration.

## Rust

- Define every dependency once in root `[workspace.dependencies]`; consumers
  inherit with `{ workspace = true }`. Sort members and dependency keys.
- Inherit workspace package settings and lints in each member.
- Keep handwritten Rust files below 600 lines, with a target of 400. Split by
  responsibility rather than compressing code or deleting useful documentation.
- Prefer type parameters at the expression (`collect::<Vec<_>>()`) when possible.
- Keep compiler internals out of the runtime dependency graph. The runtime must
  continue to compile and test with stable Rust.
- Support classic `MTL*` and Metal 4 behind the same public runtime API. Keep
  native API selection internal and check OS/device capabilities. Verify both
  paths independently on hardware before changing their execution behavior.
- Device builtins use CUDA names: `thread_idx()`, `block_idx()`, `block_dim()`,
  and `grid_dim()`, returning x/y/z coordinates. Keep them ordinary Rust functions.
- Keep the public API small and direct. Add types, helpers, options, and alternate
  paths only for a concrete current use case. Keep implementation details private.
- Prefer one clear way to perform an operation. Avoid wrapper layers and general
  abstractions introduced for hypothetical future needs.
- Use CUDA-style launch semantics: `LaunchConfig<X, Y, Z>` encodes threads per
  block as const generics; `grid: Dim3` counts runtime blocks. Each block maps
  to a Metal threadgroup. `DynamicLaunchConfig` supports runtime block shapes.
  Public kernel invocation is `Device::launch`; both paths share validation.
- Keep const block information in the configuration type. Host const generics
  do not specialize handwritten MSL. Future compiler variants and generated
  bindings must carry and enforce any specialized block-shape requirements.
- Preserve checks, assertions, address spaces, and numerical semantics. Reject
  unsupported operations explicitly; do not silently lower them differently.

## Project skills

Read the relevant repository skill before working in its area:

- `.agents/skills/metal-oxide-runtime/SKILL.md`: buffers, Metal, dispatch, safety.
- `.agents/skills/metal-oxide-compiler/SKILL.md`: rustc, device target, IR, ABI.
- `.agents/skills/metal-oxide-contributing/SKILL.md`: verification and commits.

## Verification

Run `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --locked
-- -D warnings`, and `cargo test --workspace --locked` for Rust changes.
GPU tests are explicit: `cargo test -p metal-oxide --test gpu --locked --
--ignored --test-threads=1`. State when hardware checks were not run. A skipped
GPU test is not evidence of correctness. Do not automatically run untrusted PR
code on a personal or self-hosted runner.

For compiler changes, also run tests and Clippy from `crates/metal-oxide-compiler`
with `--features rustc-private --locked --target-dir ../../target/compiler`.
That directory pins the dated nightly; the workspace root stays on stable Rust.
