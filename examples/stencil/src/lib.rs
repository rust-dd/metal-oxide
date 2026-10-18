//! Jacobi sweeps of a 5-point stencil over an f32 grid stored row by row: the kernel, its launch,
//! a CPU reference and the input values, which the example and the benchmarks run.

/// One sweep of a `width x height` grid from one buffer into another.
#[metal_oxide::kernels(math = "precise")]
pub mod five_point {
    use metal_oxide::device::{ReadBuffer, WriteBuffer, block_idx, thread_idx};

    /// Threads per block along x and along y.
    pub const TILE: u32 = 16;

    /// Interior cells take `0.25 * (((l + r) + u) + d)` of their left, right, upper and lower
    /// neighbours, boundary cells copy theirs, and only threads of z layer 0 inside the grid write.
    #[kernel(
        block = (16, 16, 1),
        requires(
            input.len() == width * height,
            output.len() == width * height,
            width * height <= 4_294_967_295
        )
    )]
    pub fn jacobi(input: ReadBuffer<f32>, output: WriteBuffer<f32>, width: u32, height: u32) {
        let (block, thread) = (block_idx(), thread_idx());
        let (x, y) = (block.x * TILE + thread.x, block.y * TILE + thread.y);
        if block.z == 0 && x < width && y < height {
            let i = y * width + x;
            // SAFETY: the launch checks that both buffers hold `width * height` values, at most
            // u32::MAX: `i` and interior neighbours are in bounds, and only this thread writes `i`.
            unsafe {
                let value = if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                    input.load_unchecked(i)
                } else {
                    let l = input.load_unchecked(i - 1);
                    let r = input.load_unchecked(i + 1);
                    let u = input.load_unchecked(i - width);
                    let d = input.load_unchecked(i + width);
                    0.25 * (((l + r) + u) + d)
                };
                output.store_unchecked(i, value);
            }
        }
    }
}

use metal_oxide::{Dim3, LaunchConfig};

/// The launch of a sweep over a `width x height` grid: a block for each 16 x 16 tile.
pub fn launch(width: u32, height: u32) -> LaunchConfig<16, 16> {
    let tiles = |cells: u32| cells.div_ceil(five_point::TILE);
    LaunchConfig::new(Dim3::xy(tiles(width), tiles(height)))
}

/// The kernel's sweep of `input` into `output`, both grids of `width x height` values, with its
/// float order.
pub fn sweep(input: &[f32], output: &mut [f32], width: usize, height: usize) {
    assert!(
        input.len() == width * height && output.len() == input.len(),
        "grids of {width} x {height}"
    );
    for y in 0..height {
        let start = y * width;
        let here = &input[start..start + width];
        let out = &mut output[start..start + width];
        let interior = y > 0 && y + 1 < height && width > 2;
        if !interior {
            out.copy_from_slice(here);
            continue;
        }
        let above = &input[start - width..start];
        let below = &input[start + width..start + 2 * width];
        out[0] = here[0];
        out[width - 1] = here[width - 1];
        for x in 1..width - 1 {
            out[x] = 0.25 * (((here[x - 1] + here[x + 1]) + above[x]) + below[x]);
        }
    }
}

/// `values`, a `width x height` grid, after `iterations` sweeps.
pub fn reference(values: &[f32], width: usize, height: usize, iterations: u32) -> Vec<f32> {
    let mut grids = [values.to_vec(), vec![0.0; values.len()]];
    for _ in 0..iterations {
        let [input, output] = &mut grids;
        sweep(input, output, width, height);
        grids.swap(0, 1);
    }
    let [result, _] = grids;
    result
}

/// `len` values in [1/16, 16) with varied exponents and mantissas. Sweeps of positive values stay
/// within their range, so they produce no subnormals, which the GPU may flush.
pub fn values(len: u32) -> Vec<f32> {
    let value = |i: u32| {
        let hash = i.wrapping_mul(2_654_435_761);
        let (exponent, mantissa) = (123 + (hash >> 29), (hash >> 6) & 0x7f_ffff);
        f32::from_bits(exponent << 23 | mantissa)
    };
    (0..len).map(value).collect()
}
