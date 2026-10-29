# Supported Rust

The compiler imports typed rustc MIR for concrete entrypoints and their reachable
helpers. Rust first checks types, lifetimes and borrowing. Kernels compile against
`core`; arbitrary host Rust and the complete `no_std` ecosystem are not supported.
GPU requirements appear in the generated [capability table](capabilities.md).

## Accepted constructs

- Scalars: `bool`, `f32`, `F16`, `BF16`, 8/16/32-bit integers, `u64`, `usize` and
  internal `isize`. Separate kernel crates can also use unstable primitive `f16`.
- Integer powers, checked division/remainder, byte swaps, `cmp`/`partial_cmp`,
  derived ordering and `signum`; integer and float `clamp`, plus `f32::copysign`,
  `signum` and `total_cmp`. Invalid clamp bounds remain checked, including NaN
  bounds on `f32`.
- Records, tuples, arrays, payload enums with default discriminants, `Option`,
  `Result`, exhaustive matches and `?`. Enums are internal values, not ABI parameters.
- References, slices and lifetime-bearing aggregates inside kernels and helpers.
  References retain their device, threadgroup or thread-local address space.
- Concrete generic type/const instances, trait calls with a known implementation,
  closures and supported core iterators. Generic entrypoints declare their concrete
  `instances(...)`; lifetime parameters need no instance argument.
- Checked slice indexing, assertions and literal panics. Checks that cannot be
  proved become GPU failure checks where participation permits them.
- Supported f32 math, including exercised `libm` and `num-traits` workloads.
  `dependencies(...)` declares `no_std` kernel dependencies, which build with the
  features and `default-features` setting that their package declares; it does not make
  all of a dependency's operations available.

Only owned ABI values and buffer handles cross a kernel entry boundary. Host-shared
records use canonical layout codecs rather than arbitrary host memory layout.
Unsupported layouts, conversions or MIR constructs produce diagnostics.

## Atomics and cooperation

`u32`/`i32` atomics support device and threadgroup load, store, exchange, weak
compare/exchange, arithmetic, min/max and bitwise operations. `f32` supports device
load, store, exchange, weak compare/exchange and add/subtract; compare/exchange
compares bits. `u64` supports device min/max without returning the previous value.
Orderings must be constant; `SeqCst` and unsupported operation/address-space
combinations are rejected. Acquire/release requires MSL 4.1 and macOS 27 and still
has no hardware validation evidence.

SIMD reductions, scans, shuffles, votes and barriers require their documented
participation rules. `simdgroup::Matrix<T>` represents an 8x8 matrix distributed
across 32 lanes, with f32/F16/BF16 types and supported mixed accumulation. Matrix
kernels declare a block shape whose thread count is a multiple of 32. Operands
and control must be uniform within each SIMD group; reference-read operands are
treated as lane-varying. Loads/stores remain unsafe.

Within one thread, `core::simd` supports 2–4 lanes of f32 and 8/16/32-bit integers:
construction, lane access, arithmetic, masks, comparisons, selection, reductions,
casts, swizzles, saturation and bit counts lower to scalar operations. These
vectors do not cross the ABI. Host files using this unstable API need nightly;
separate kernel crates already use the pinned compiler toolchain.

## Rejected constructs

Recursion, indirect calls through function pointers or trait objects, destructors,
coroutines, `f64`, `i64` and 128-bit integers are rejected. Dynamic strings and trait
objects have no device representation; literal panic/expect messages are
specialized during compilation. Reinterpreting pointer casts, unsupported enum
layouts, wider SIMD vectors, gather/scatter and masked vector loads are rejected.
Heap allocation, host I/O and operating-system facilities are outside the device
subset. A varying branch or early return that bypasses required cooperative work
is rejected even when the corresponding scalar code would otherwise compile.

Kernels always compile with the release profile, even when the host uses a debug
build. Integer overflow checks are off by default. To enable them, add this to
the workspace `Cargo.toml`:

```toml
[profile.release]
overflow-checks = true
```

Bounds, division and explicit panics remain checked regardless of this setting.
Numerical transformations follow the explicit [math mode](performance.md#numerical-modes).
