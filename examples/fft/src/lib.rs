//! Batched forward FFT of complex f32 rows in split real and imaginary arrays: the Stockham
//! kernel, the plan of its stages and an f64 reference, which the example and the benchmarks run.

/// Radix-2 Stockham stages over rows of `n` values, a power of two of at least 2: after the
/// stages of half span 1, 2, ..., `n / 2`, each row holds its forward transform in natural order.
#[metal_oxide::kernels(math = "precise")]
pub mod stockham {
    use core::f32::consts::PI;
    use metal_oxide::device::{ReadBuffer, WriteBuffer, block_idx, thread_idx};

    /// Butterflies per block, one for each thread.
    pub const BLOCK: u32 = 256;

    /// # Safety
    /// Every buffer holds `n` values, a power of two, per row of a 2-D grid's y axis, at most 2^32
    /// in all, and nothing else accesses them; `half_span` is a power of two below `n`.
    #[kernel(block = (256, 1, 1))]
    pub unsafe fn fft_stage(
        input_re: ReadBuffer<f32>,
        input_im: ReadBuffer<f32>,
        output_re: WriteBuffer<f32>,
        output_im: WriteBuffer<f32>,
        n: u32,
        half_span: u32,
    ) {
        let half = n >> 1;
        let butterfly = block_idx().x * BLOCK + thread_idx().x;
        if butterfly < half {
            let row = block_idx().y * n;
            let j = butterfly & (half_span - 1);
            let (sin, cos) = (-2.0 * PI * j as f32 / (half_span << 1) as f32).sin_cos();
            let (input, output) = (row + butterfly, row + (butterfly << 1) - j);
            // SAFETY: with powers of two, the inputs and outputs lie in the butterfly's row; only
            // this butterfly writes these outputs, and nothing else accesses the buffers.
            unsafe {
                let a_re = input_re.load_unchecked(input);
                let a_im = input_im.load_unchecked(input);
                let b_re = input_re.load_unchecked(input + half);
                let b_im = input_im.load_unchecked(input + half);
                let t_re = b_re * cos - b_im * sin;
                let t_im = b_re * sin + b_im * cos;
                output_re.store_unchecked(output, a_re + t_re);
                output_im.store_unchecked(output, a_im + t_im);
                output_re.store_unchecked(output + half_span, a_re - t_re);
                output_im.store_unchecked(output + half_span, a_im - t_im);
            }
        }
    }
}

pub mod cpu;

use metal_oxide::{Dim3, LaunchConfig};

/// The half spans of the stages that transform rows of `n` values, a power of two of at least 2,
/// in order; each stage reads the buffer pair that the previous one wrote.
pub fn half_spans(n: u32) -> impl Iterator<Item = u32> {
    assert!(n >= 2 && n.is_power_of_two(), "{n} values per row");
    (0..n.trailing_zeros()).map(|stage| 1 << stage)
}

/// The launch of every stage over `rows` rows of `n` values: x counts blocks of a row's `n / 2`
/// butterflies, y counts rows.
pub fn launch(n: u32, rows: u32) -> LaunchConfig<256> {
    LaunchConfig::new(Dim3::xy((n / 2).div_ceil(stockham::BLOCK), rows))
}
