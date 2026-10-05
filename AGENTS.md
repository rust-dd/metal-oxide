# metal-oxide

Build a Metal-first Rust compute compiler for macOS on Apple Silicon. Follow
[docs/roadmap.md](docs/roadmap.md) and [docs/architecture.md](docs/architecture.md).
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

## Rust

- Define every dependency once in root `[workspace.dependencies]`; consumers
  inherit with `{ workspace = true }`. Sort members and dependency keys.
- Inherit workspace package settings and lints in each member.
- Keep handwritten Rust files below 600 lines, with a target of 400. Split by
  responsibility rather than compressing code or deleting useful documentation.
- Prefer type parameters at the expression (`collect::<Vec<_>>()`) when possible.
- Keep compiler internals out of the runtime dependency graph. The runtime must
  continue to compile and test with stable Rust.
- Use CUDA-style launch semantics: `LaunchConfig.grid` counts blocks and
  `LaunchConfig.block` counts threads per block. Both use `Dim3`; each block maps
  to a Metal threadgroup. Public kernel invocation is `Device::launch`.
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
