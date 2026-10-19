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
python3 scripts/bench.py --samples 20 --warmup 10 --output target/benchmarks/baseline.json
```

`python3 scripts/test_bench.py` checks the driver itself. The benchmark compares vec-add,
reduction, scan, histogram, particle update, matmul, a bitonic sort of u32 keys (`--sort
KEYS`, 2^23 by default), a batched Stockham FFT (`--fft N ROWS`, 8192 rows of 4096 complex
values by default), a 5-point Jacobi stencil (`--stencil WIDTH HEIGHT ITERATIONS`, 2048
sweeps of a 2048 x 2048 grid by default) and the f32 and f16 SIMD-group-matrix GEMM (`--gemm
M N K`, 1024 cubed by default) with CPU results and matching handwritten MSL on both
backends; the classic run also measures the GEMM with MetalPerformanceShaders as backend
`mps`, and both backends measure Metal Performance Primitives' `matmul2d` as variant `mpp`
(`benchmarks/reference-msl/gemm-mpp.metal`, MSL 4.0). Where the compiler has no Metal
tensors, or the device lacks the Metal 4 family or rejects the library, the report records
why `mpp` was skipped and the other candidates run. `--cases LIST` builds and measures only
the listed workloads (all by default; `cooperation` holds histogram and scan). Sort and
stencil results must equal the CPU's bitwise; FFT results must lie within 4e-6 * log2(n) *
max(1, rms of their row) of an f64 reference, and both FFT variants must agree bitwise. A
GEMM sample runs `--gemm-repeat R` GEMMs (64 by default) back to back in one command buffer,
so the GPU is not idle between timed GEMMs; `gpu_per_gemm_ns` is `gpu_ns / R` and
`host_per_gemm_ns` is `(submit_ns + wait_ns) / R`. Within a runner process the candidates of
every case take turns sample by sample (rust and msl; for the GEMM also mpp and, in the
classic run, mps; the first one rotates every round). Classic and Metal 4 run in separate
processes and MLX afterwards in its own, so MLX and cross-backend ratios carry drift between
runs: repeat runs and report the spread. The rust and msl kernels share their hot loops, so
a rust/msl ratio near 1 confirms the interleave but cannot reveal drift between runs. The
JSON report separates build/rebuild, pipeline, host copy/upload/encoding/wait/readback and
native GPU times. Every sample is checked; unavailable GPU timestamps are null, and so is
the CPU time of GEMM, whose exact result is computed once per shape. Submit time measures
the first poll with prepared pipelines. Under saturation, later polls may submit the
command. Host wait includes preparation, execution and scheduling, so it does not measure
queue latency alone.

`--backends classic` (or `metal4`) measures only the listed backends, for a Mac without
Metal 4; MPS runs with classic only. `--mlx` also runs the GEMM with MLX in
`target/benchmarks/mlx-venv`, which the driver creates and pins to `mlx==0.32.4`, and adds
it under the report's `mlx` key. MLX runs R independent matmuls per sample, back to back,
and checks them against the CPU's exact result before the warmup and after the last sample,
since checks between samples were observed to double its times. It exposes no GPU timestamps
and returns an f16 C for f16 inputs, so compare its `host_per_gemm_ns` with the other
backends' `host_per_gemm_ns`, not with GPU times. The report's `machine` records the chip,
model, memory and GPU core count beside the OS and toolchain versions; a value the host
cannot report is null.

Measured on an M4 Max: compare GEMMs at the default R = 64 with `--warmup 10`; single runner
runs at R = 8 measured 0-31 % slower for rust and msl at 1024 cubed while MPS did not move.
MPS's host time includes its own encoding (about 14 us per GEMM) and ran 14-28 us per GEMM
above its GPU time at 1024 cubed and 21-36 us at 2048 cubed, so compare MPS on
`gpu_per_gemm_ns`; the first MPS case's pipeline time includes MPS's library load. MLX's
matmuls are independent and may overlap, unlike the runner's GEMMs, which chain through one
C; a dependent MLX chain measured within 1-5 %. Absolute times drifted by up to 2x between
runs while the GPU was shared with other applications, so compare candidates within one run
and repeat runs. The kernel's direct path needs k to be a multiple of 16, so a shape such as
1000 cubed measures its staged path. The gemm case needs an Apple7 or later GPU (SIMD-group
matrices); elsewhere the runner reaches it only after every other case, minutes at the
defaults, then stops with `UnsupportedDevice` and bench.py writes no report, so leave gemm
out of `--cases` there. `--mlx` downloads mlx from PyPI on first use. Very long command
buffers (a large R times a large shape) may hit the GPU's interactivity limit and fail as a
command-buffer error.

Measured on an M4 Max for the other cases: each sample first runs its serial CPU reference
(about 0.09 s for the sort, 0.3 s for the FFT and 1.1 s for the stencil at their defaults)
while the GPU idles, so shorter GPU bursts pair poorly (256 stencil sweeps swung between 12
and 27 ms); the sort, FFT and stencil defaults give about 25, 20 and 100 ms of GPU work per
sample. A default run of every case takes about four minutes, and the FFT's runner process
peaks at about 2.6 GB. Sample i of each candidate comes from the same round, so compare rust
and msl by the median of per-round ratios: in three default runs (the sort then at
2^22 keys) it stayed within 0.95-1.10 for every case and run, while one ratio of medians
reached 1.70 at a per-round 1.01. Small cases' GPU times jumped between levels about 3x
apart from one process to the next (vec-add 0.029 or 0.096 ms on Metal 4), so their backend
ratios are not meaningful: the FFT at 4096 x 256 measured Metal 4 at 0.46-2.3x classic in
separate processes and 1.00-1.04x with both backends alternating in one process. A dependent
dispatch costs about 0.5 us more on Metal 4 than on classic (2.0 against 1.5 us per sweep of
a 64 x 64 grid), at the encoder barrier between them, and on classic a generated kernel's
gate test, a device load before its first load, adds 0.10-0.15 us per dispatch; at the
default sizes the barrier costs about 1 % of a stencil sweep and the gate up to 2 %, and the
FFT ran about 10 % faster on Metal 4.
