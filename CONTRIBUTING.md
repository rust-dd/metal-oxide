# Contributing

Read [AGENTS.md](AGENTS.md) and the relevant skill under `.agents/skills` before
changing code. M0 runtime, M1 frontend, and M2 IR/MSL lowering are implemented;
M3 artifact integration is next in [the roadmap](docs/roadmap.md).

## Workspace

From the repository root, run:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo test --package metal-oxide --test gpu --locked -- --ignored --test-threads=1
```

Ordinary tests leave hardware checks ignored. Keep dependencies centralized and
sorted in root `Cargo.toml`; every consumer uses `workspace = true`. Commit
`Cargo.lock`. Keep the runtime buildable with stable Rust and independent of the
nightly compiler.

Add tests for observable behavior and safety boundaries. For runtime changes,
also execute the ignored hardware tests on an Apple Silicon machine. Record
which checks actually ran and their environment. CI compilation is not evidence
that a GPU kernel executed correctly.

Run the compiler checks from `crates/metal-oxide-compiler`, where the dated nightly
is pinned:

```sh
cargo test --features rustc-private --locked --target-dir ../../target/compiler
cargo clippy --features rustc-private --all-targets --locked --target-dir ../../target/compiler -- -D warnings
```

The compiler tests build matching device `core`/`compiler_builtins` metadata,
the host marker macro, the device crate, and separate kernels. Portable tests
check MIR/IR diagnostics and execute the emitted MSL subset with `clang++` and
a test-only Metal header shim. GPU execution requires the explicit command:

```sh
cargo test --features rustc-private --test gpu --locked --target-dir ../../target/compiler -- --ignored --test-threads=1
```

These hardware tests load the exact compiler output through the runtime and
compare results with CPU references. Unsupported Rust operations must fail with
diagnostics. Do not discard MIR assertions or change numerical semantics to
make a kernel compile.

## API design

Keep the public API small and direct, with one clear path for each operation.
Every new type, helper, option, or alternative path needs a concrete current use
case. Keep implementation helpers private and defer abstractions for hypothetical
future needs.

## Commits and prose

Use concise Conventional Commit subjects, for example:

```text
feat: add synchronous Metal dispatch
fix: reject oversized buffer allocations
chore: initialize workspace
docs: define device memory contracts
```

Commits and PRs describe the change and relevant verification. Do not append
co-author trailers, AI attribution, session metadata, tool transcripts, or
generated-by footers.

Code comments explain a safety invariant or a non-obvious reason. Retain useful
rustdoc and `SAFETY` comments. Avoid decorative banners, separator lines,
edit-history narration, commented-out code, and comments that repeat a statement.
Release history belongs in commits and release notes, not Rust doc comments.
Use clear names and small functions to express intent. Rustdoc covers observable
behavior and caller obligations; omit trivial explanations and implementation
narration.

## Hardware and CI

Keep portable checks separate from explicit GPU checks. Use a hardware runner
only after verifying its actual Metal device. Do not automatically execute
untrusted PR code on a personal or self-hosted runner. No hardware runner is
configured in this repository yet.
