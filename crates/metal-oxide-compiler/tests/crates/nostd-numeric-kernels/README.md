# no_std numeric acceptance

This crate compiles the registry releases of `libm` 0.2.16 and `num-traits`
0.2.19 through their Rust MIR. `num-traits` uses its `libm` feature with `std`
disabled. The compiler does not replace these library methods with intrinsics.

The accepted workloads cover `libm`'s `fabsf`, `copysignf`, `modff`, `floorf`,
`ceilf`, `roundf` and `truncf`; generic `Float` rounding, absolute value and
integer powers; and generic `PrimInt` checked powers, rotations and saturating
addition/subtraction for i8/i16/i32/u8/u16/u32/u64. The compiler's `nostd` test
binary compares emitted MSL and GPU results with these functions on the CPU.

This is a concrete subset, not whole-library compatibility. The
`unsupported-f64` feature exercises `libm::cbrtf`, whose f64 intermediates are
rejected. `unsupported-sine` exercises `libm::sinf`, which reaches unsupported
volatile loads and foreign calls in argument reduction. Ordinary workspace
builds leave both rejection features disabled.
