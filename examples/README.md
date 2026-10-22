# Examples

Run these commands from the repository root on macOS 15+ with Apple Silicon.
Follow the [installation guide](https://rust-dd.github.io/metal-oxide/installation.html)
for Xcode and the Metal Toolchain, then set up the checkout's kernel compiler:

```sh
cargo run -p cargo-metal --locked -- setup
```

Each command runs the example's result checks. `--release` also optimizes the host
and its CPU references; kernels always use the release profile.

| Example | What it demonstrates | Run |
| --- | --- | --- |
| [vec-add](vec-add/src/main.rs) | A kernel beside its host and safe per-thread output access | `cargo run -p vec-add --release --locked` |
| [saxpy](saxpy/src/main.rs) | A safe kernel with a buffer-length launch contract | `cargo run -p saxpy --release --locked` |
| [reduction](reduction/src/main.rs) | Block reduction using shared memory and barriers | `cargo run -p reduction --release --locked` |
| [matmul](matmul/host/src/main.rs) | A separate kernel crate, tiled multiplication and SIMD reductions | `cargo run -p matmul --release --locked` |
| [gemm](gemm/host/src/main.rs) | f32 and F16 multiplication using SIMD-group matrices | `cargo run -p gemm --release --locked` |
| [cooperation](cooperation/host/src/main.rs) | Atomic histogram, hierarchical scan and stream compaction | `cargo run -p cooperation --release --locked` |
| [pipeline](pipeline/src/main.rs) | Async kernel chains with shared types and helpers | `cargo run -p pipeline --release --locked` |
| [particle-update](particle-update/src/lib.rs) | Shared records, generic kernels and CPU/MSL comparisons | `cargo run -p particle-update --release --locked` |
| [sort](sort/src/lib.rs) | Bitonic sorting through shared memory and batched launches | `cargo run -p sort --release --locked` |
| [fft](fft/src/lib.rs) | Batched Stockham FFT with separate real and imaginary buffers | `cargo run -p fft --release --locked` |
| [stencil](stencil/src/lib.rs) | Jacobi sweeps of a 5-point stencil over an f32 grid | `cargo run -p stencil --release --locked` |
| [spectral](spectral/README.md) | FFT, fractional Gaussian noise, sheet transforms and Euler sampling | `cargo run -p spectral --release --locked` |

Set `METAL_OXIDE_BACKEND=classic` or `METAL_OXIDE_BACKEND=metal4` before a command
to select a native API. Metal 4 requires a compatible OS and device. The GEMM
example reports unsupported SIMD-group matrices and exits on devices without
that capability; this is not a successful GPU comparison.

The [spectral guide](spectral/README.md) also covers fast math and benchmarks.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full hardware test suite and
benchmark commands.
