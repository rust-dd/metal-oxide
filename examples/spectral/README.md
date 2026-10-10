# Spectral kernels

Separate `no_std` Rust kernels exercise general compiler/runtime features using
FFT, fractional Gaussian noise, sheet transforms and Euler sampling patterns.

```sh
METAL_OXIDE_BACKEND=classic cargo run -p cargo-metal -- run -p spectral --release
METAL_OXIDE_BACKEND=metal4 cargo run -p cargo-metal -- run -p spectral --release --math-mode fast
python3 scripts/bench-spectral.py --samples 20 --warmup 5
```

Validation compares generated fGN with independent f64 CPU transforms and native
MSL through 65536 output points. It checks seeded u64 streams, odd and wrapping
chunk boundaries, Hurst covariance, and a native fGN allocation consumed in place
by generated Euler code. Sheet passes check row/column transforms, transpose,
crop/correction and chunk invariance. Positive test eigenvalues exercise the
sheet device passes; this is not a complete fractional-sheet host implementation.

Euler checks cover Knuth counts, two Marsaglia–Tsang gamma draws, Gaussian jump
sizes, all sixteen coefficient opcodes, 512-element history, 176-element lift,
series bucketing and table interpolation. Native reference sampling preserves
the stream; the CPU count reference models Metal's subnormal flushing.

Benchmarks cover 4096–65536 points and batches through 16384 rows, using the same
tile plan and numerical mode. Native fGN uses raw classic Metal, `setBytes`, and
separate encoders for successive passes. Generated fGN uses the public runtime
on each backend. Buffers are reused; large batches stream through bounded chunks
derived from the device budget. Reports distinguish logical rows from allocated
chunk rows and separate build/cache, setup, allocation/upload, pipeline, encode,
submit, wait, GPU and CPU-view costs. Samples alternate execution order and check
the final chunk against native output. Setup and allocation costs are reported
separately from repeated end-to-end execution. Ratios are measurements, not an
unconditional performance guarantee.

For cases with at most sixteen rows, each sample repeats the complete logical
run sixteen times to reduce scheduling noise. Other cases use one repetition.
The report records this count; sample times are totals over those repetitions.
