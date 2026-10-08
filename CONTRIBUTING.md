# Contributing

Read [AGENTS.md](AGENTS.md) and the relevant skill under `.agents/skills` before
changing code.

## Workspace

From the repository root, run:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
bash scripts/test-gpu.sh
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

Compiler tests require `clang++`. They build matching device
`core`/`compiler_builtins` metadata, the host marker macro, the device crate,
and separate kernels. Portable tests
check MIR/IR diagnostics and execute the emitted MSL subset with `clang++` and
a test-only Metal header shim.

These hardware tests select classic Metal and Metal 4 independently, load the
exact compiler output, and compare results with CPU and MSL references. They also
test async kernel chains, cancelled submissions, and generated bindings.
Unsupported Rust operations must fail with diagnostics. Do not discard MIR
assertions or change numerical semantics to
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

Verify a private bundle with the independent host/kernel/helper workspace:

```sh
python3 scripts/test-installed.py target/dist/metal-oxide-0.1.0-alpha.1-aarch64-apple-darwin.tar.gz
```

This runs both backends, debug/release builds, and a relocated stable host with
only the runtime crates and precompiled bindings/artifact. It retains logs under
`target/installed-*`. Portable packaging and benchmark-driver checks run with
`python3 scripts/test_package.py` and `python3 scripts/test_bench.py`.

Keep portable checks separate from explicit GPU checks. Use a hardware runner
only after verifying its actual Metal device. Do not automatically execute
untrusted PR code on a personal or self-hosted runner. No hardware runner is
registered yet. The manual GPU workflow runs only `main` on a verified runner
labelled `metal-oxide-gpu`; it has no pull-request trigger.

## Private installation

Package a clean checkout and install into a fresh versioned directory
(Python 3.11+):

```sh
python3 scripts/package.py build
python3 scripts/package.py install target/dist/metal-oxide-0.1.0-alpha.1-aarch64-apple-darwin.tar.gz --prefix "$HOME/.local/metal-oxide/0.1.0-alpha.1"
export PATH="$HOME/.local/metal-oxide/0.1.0-alpha.1/bin:$PATH"
cargo metal doctor
```

The bundle includes the CLI, pinned compiler, source crates, target data,
licenses and checksums. Use the runtime and device crates under the installed
`source/crates` as path dependencies. The CLI verifies the adjacent compiler;
`METAL_OXIDE_COMPILER` selects an explicit compiler instead.
A compiled host needs `manifest.json` and `kernels.metallib` at runtime.

## Benchmarks

```sh
python3 scripts/bench.py --samples 20 --warmup 5 --output target/benchmarks/baseline.json
```

Compares vec-add, reduction, scan, histogram, particle update and matmul with
CPU results and matching handwritten MSL on both backends. The JSON report
separates build/cache, pipeline, host copy/upload/encoding/wait/readback and
native GPU times. Every sample is checked; unavailable GPU timestamps are null.
Submit time includes encoding and may overlap execution. Host wait includes
execution and scheduling, so it does not measure queue latency alone.
