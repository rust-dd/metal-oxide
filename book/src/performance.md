# Performance

Measure the workload on the actual device. First-use compilation, host encoding,
CPU views and GPU execution are separate costs. `operation.timed()` returns GPU
time with the result; benchmark reports record unavailable timestamps as null.
Use [inspection](debugging.md#inspect-generated-code) to compare AIR when changing
kernel hot paths.

Use `sequence` to amortize command submission for several operations. Keep
intermediate data on the GPU and read back after the chain. Choose block sizes
within pipeline and threadgroup-memory limits, and measure alternatives; a host
const block shape does not by itself specialize shader code.

## Numerical modes

The default precise mode disables implicit floating-point contraction.
`#[kernels(math = "fast")]` explicitly permits fast functions, reassociation,
reciprocal approximations and contraction. Arithmetic retains NaN and infinity
support; fast functions have narrower accuracy/domain guarantees and zero signs
are unspecified. The mode is part of the saved artifact. Compare equivalent modes
and numerical error, rather than assuming that fast math improves throughput.

## Repository benchmarks

From the checkout:

```sh
python3 scripts/bench.py --samples 20 --warmup 10 --output target/benchmarks/baseline.json
```

The driver checks results against CPU references and matching handwritten MSL. It
covers vector addition, reduction, scan, histogram, particles, matrix multiply,
sort, FFT, stencil and SIMD-group GEMM. It separates build, pipeline, host and GPU
costs. `--backends classic` restricts a run to classic Metal; `--cases LIST` selects
workloads. The GEMM needs an Apple7 family GPU, which the hosted Paravirtual device
does not report; omit GEMM there.

Candidates interleave within a backend's process; compare per-round ratios and
repeat runs because other GPU activity and thermal state affect timings. Classic
and Metal 4 run in separate processes. CPU reference work between samples can also
leave the GPU idle. A command buffer containing too much work can exceed the GPU's
interactivity limit.

GEMM comparisons include MPS on classic and MPP `matmul2d` when available. MPP is a
benchmark reference, with implementation selected by the OS; it is not a Rust
tensor API. The M4 Max measurements did not meet a 1.20 Rust/MPP GPU-time threshold
for both input types on both backends in every repeated run. No parity claim
extends to other GPU generations. `--mlx` adds a separately installed reference;
its host timings and result precision require care when comparing.

For the measurement protocol and recorded limitations, see the repository's
[benchmark guide](https://github.com/rust-dd/metal-oxide/blob/main/CONTRIBUTING.md#benchmarks).
