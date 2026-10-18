# Contributing

Read [AGENTS.md](AGENTS.md) and the relevant skill under `.agents/skills` before
changing code.

## Workspace

From the repository root, run:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo +1.89.0 check --workspace --all-targets --locked
bash scripts/test-gpu.sh
```

`scripts/test-gpu.sh` runs the hardware suites through
[cargo-nextest](https://nexte.st) (`cargo install cargo-nextest --locked`). CI runs the
portable suites with `cargo nextest run --profile ci` and the documentation examples with
`cargo test --doc`; the `ci` and `gpu` profiles in `.config/nextest.toml` print each finished
test with a running `(N/total)` counter.

Ordinary tests leave hardware checks ignored. The examples keep their saved artifacts
in `metal/` beside their sources, and the stable CI job builds from them with
`METAL_OXIDE_ARTIFACTS=locked`. A change to a kernel or to `metal-oxide-device` makes
them stale: `cargo build --workspace` and `cargo build -p spectral --features fast`
rewrite them; commit the result. Keep dependencies centralized and
sorted in root `Cargo.toml`; every consumer uses `workspace = true`. The minimum
supported stable Rust is 1.89; development uses the pinned 1.99 toolchain. Commit
`Cargo.lock`. Keep the runtime buildable with stable Rust and independent of the
nightly compiler.

Add tests for observable behavior and safety boundaries. For runtime changes,
also execute the ignored hardware tests on an Apple Silicon machine. Record
which checks actually ran and their environment. CI compilation is not evidence
that a GPU kernel executed correctly. Changes to cross-thread runtime state also
run its loom models:

```sh
RUSTFLAGS="--cfg loom" cargo test -p metal-oxide --lib --release --locked --target-dir target/loom loom
```

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
test async kernel chains, cancelled operations, and generated launch functions.
The numerical-mode test compares source, offline and handwritten MSL results,
then measures matching Box-Muller/Euler workloads. It writes raw GPU timing
samples to `target/math-modes-{classic,metal4}.json`; these measurements do not
imply a speedup from fast math or consumer performance parity.
The embedded-library test builds 484 entrypoints, builds its host from the saved
artifact on Rust 1.89 without the kernel toolchain, and runs it relocated with an
empty environment. It checks lazy loading, ABI/access/shape validation and corrupt
or swapped library bytes on both native backends.
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

A commit message is one subject line without a body. PRs describe the change
and relevant verification. Do not append co-author trailers, AI attribution,
session metadata, tool transcripts, or generated-by footers.

Code comments explain a safety invariant or a non-obvious reason. Retain useful
rustdoc and `SAFETY` comments. Avoid decorative banners, separator lines,
edit-history narration, commented-out code, and comments that repeat a statement.
Release history belongs in commits and release notes, not Rust doc comments.
Use clear names and small functions to express intent. Rustdoc covers observable
behavior and caller obligations; omit trivial explanations and implementation
narration.

## Hardware and CI

Hosted CI runs the workspace and compiler checks above on ubuntu-24.04, macos-15
and macos-26. On macOS it also installs cargo-metal, runs `cargo metal setup` and
tests `tests/installed`, an application outside the workspace, with plain Cargo.
The macOS runners are M1 virtual machines whose Apple Paravirtual device lacks the
Metal 4 family, so CI runs `bash scripts/test-gpu.sh classic` there; verify Metal 4
on a Mac. Keep portable checks separate from explicit GPU
checks. Use a hardware runner only after verifying its actual Metal device. Do
not automatically execute untrusted PR code on a personal or self-hosted runner.
No self-hosted runner is registered yet. The manual GPU workflow runs only
`main` on a verified runner labelled `metal-oxide-gpu`; it has no pull-request
trigger.

## Benchmarks

```sh
python3 scripts/bench.py --samples 20 --warmup 5 --output target/benchmarks/baseline.json
```

`python3 scripts/test_bench.py` checks the driver itself. The benchmark compares
vec-add, reduction, scan, histogram, particle update, matmul and the f32 and f16
SIMD-group-matrix GEMM (`--gemm M N K`, 1024 cubed by default) with CPU results and
matching handwritten MSL on both backends; the classic run also measures the GEMM with
MetalPerformanceShaders as backend `mps`. A GEMM sample runs `--gemm-repeat R` GEMMs
(8 by default) back to back in one command buffer, so the GPU is not idle between timed
GEMMs; `gpu_per_gemm_ns` is `gpu_ns / R`. The JSON report
separates build/rebuild, pipeline, host copy/upload/encoding/wait/readback and
native GPU times. Every sample is checked; unavailable GPU timestamps are null, and so
is the CPU time of GEMM, whose exact result is computed once per shape.
Submit time measures the first poll with prepared pipelines. Under saturation, later
polls may submit the command. Host wait includes preparation, execution and scheduling,
so it does not measure queue latency alone.
