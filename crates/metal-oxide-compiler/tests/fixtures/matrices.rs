#![no_std]
use metal_oxide_device::{
    BF16, DynamicThreadgroupBuffer, F16, ReadBuffer, ReadWriteBuffer, WriteBuffer, block_idx,
    kernel,
    simdgroup::{self, Matrix},
    thread_idx, threadgroup,
};

macro_rules! round_trip {
    ($name:ident, $ty:ty) => {
        // Output rows 16 * path + 8 * group hold path `path` of each SIMD group's block.
        #[kernel(block = (64, 1, 1))]
        pub unsafe fn $name(
            input: ReadBuffer<$ty>,
            output: WriteBuffer<$ty>,
            scratch: DynamicThreadgroupBuffer<$ty>,
        ) {
            let shared = threadgroup::shared::<$ty, 256>();
            let row = simdgroup::group_id() * 8;
            // SAFETY: every block lies in its buffer and a barrier orders the threadgroup copies.
            unsafe {
                let block = input.load_matrix(row, 0, 8);
                output.store_matrix(row, 0, 8, block);
                output.store_matrix(16 + row, 0, 8, input.load_matrix_transposed(row, 0, 8));
                output.store_matrix_transposed(32 + row, 0, 8, block);
                shared.store_matrix(row, 0, 8, block);
                scratch.store_matrix_transposed(row, 0, 8, block);
                shared.store_matrix_transposed(16 + row, 0, 8, block);
                scratch.store_matrix(16 + row, 0, 8, block);
                threadgroup::barrier();
                output.store_matrix(48 + row, 0, 8, shared.load_matrix(row, 0, 8));
                output.store_matrix(64 + row, 0, 8, scratch.load_matrix(row, 0, 8));
                let flipped = scratch.load_matrix_transposed(16 + row, 0, 8);
                output.store_matrix(80 + row, 0, 8, flipped);
                output.store_matrix(96 + row, 0, 8, shared.load_matrix(16 + row, 0, 8));
            }
        }
    };
}
round_trip!(round_trip_f32, f32);
round_trip!(round_trip_f16, F16);
round_trip!(round_trip_bf16, BF16);

macro_rules! multiply {
    ($name:ident, $left:ty, $right:ty, $accumulator:ty) => {
        #[kernel(block = (32, 1, 1))]
        pub unsafe fn $name(
            a: ReadBuffer<$left>,
            b: ReadBuffer<$right>,
            c: ReadBuffer<$accumulator>,
            d: WriteBuffer<$accumulator>,
        ) {
            // SAFETY: each buffer holds one 8x8 matrix.
            unsafe {
                let product = Matrix::multiply_accumulate(
                    a.load_matrix(0, 0, 8),
                    b.load_matrix(0, 0, 8),
                    c.load_matrix(0, 0, 8),
                );
                d.store_matrix(0, 0, 8, product);
            }
        }
    };
}
multiply!(multiply_f32, f32, f32, f32);
multiply!(multiply_f16, F16, F16, f32);
multiply!(multiply_bf16, BF16, BF16, f32);
multiply!(multiply_f16_accumulator, F16, F16, F16);
multiply!(multiply_bf16_accumulator, BF16, BF16, BF16);
multiply!(multiply_f16_bf16, F16, BF16, f32);
multiply!(multiply_f32_f16, f32, F16, f32);

// C = transpose(C * B + C), computed in place in one 8x8 block.
#[kernel(block = (32, 1, 1))]
pub unsafe fn in_place_f32(b: ReadBuffer<f32>, c: ReadWriteBuffer<f32>) {
    // SAFETY: each buffer holds one 8x8 matrix and the loads precede the store.
    unsafe {
        let loaded = c.load_matrix(0, 0, 8);
        let product = Matrix::multiply_accumulate(loaded, b.load_matrix(0, 0, 8), loaded);
        c.store_matrix_transposed(0, 0, 8, product);
    }
}

fn accumulate(
    acc: [Matrix<f32>; 2],
    a: Matrix<F16>,
    b: (Matrix<F16>, Matrix<F16>),
) -> [Matrix<f32>; 2] {
    [
        Matrix::multiply_accumulate(a, b.0, acc[0]),
        Matrix::multiply_accumulate(a, b.1, acc[1]),
    ]
}

macro_rules! gemm {
    ($name:ident, $transposed:literal) => {
        // A 16x16 tile of C = A * B per block, B being k x n, or n x k when transposed.
        #[kernel(block = (64, 1, 1))]
        pub unsafe fn $name(
            a: ReadBuffer<F16>,
            b: ReadBuffer<F16>,
            c: WriteBuffer<f32>,
            m: u32,
            n: u32,
            k: u32,
        ) {
            let tile_a = threadgroup::shared::<F16, 128>();
            let tile_b = threadgroup::shared::<F16, 128>();
            let tile_c = threadgroup::shared::<f32, 256>();
            let thread = thread_idx().x;
            let group = simdgroup::group_id();
            let (row0, column0) = (block_idx().y * 16, block_idx().x * 16);
            let zero = F16::from_f32(0.0);
            let mut acc = [Matrix::splat(0.0_f32); 2];
            let mut base = 0;
            // SAFETY: staging covers each tile once, barriers order it, and C tiles are disjoint.
            unsafe {
                while base < k {
                    let mut slot = thread;
                    while slot < 128 {
                        let (row, inner) = (row0 + slot / 8, base + slot % 8);
                        let a_value = if row < m && inner < k {
                            a.load_unchecked(row * k + inner)
                        } else {
                            zero
                        };
                        tile_a.store_unchecked(slot, a_value);
                        let b_value = if $transposed {
                            let column = column0 + slot / 8;
                            if column < n && inner < k {
                                b.load_unchecked(column * k + inner)
                            } else {
                                zero
                            }
                        } else {
                            let (inner, column) = (base + slot / 16, column0 + slot % 16);
                            if column < n && inner < k {
                                b.load_unchecked(inner * n + column)
                            } else {
                                zero
                            }
                        };
                        tile_b.store_unchecked(slot, b_value);
                        slot += 64;
                    }
                    threadgroup::barrier();
                    let left = tile_a.load_matrix(group * 8, 0, 8);
                    let right = if $transposed {
                        (
                            tile_b.load_matrix_transposed(0, 0, 8),
                            tile_b.load_matrix_transposed(8, 0, 8),
                        )
                    } else {
                        (tile_b.load_matrix(0, 0, 16), tile_b.load_matrix(0, 8, 16))
                    };
                    acc = accumulate(acc, left, right);
                    threadgroup::barrier();
                    base += 8;
                }
                if row0 + 16 <= m && column0 + 16 <= n {
                    c.store_matrix(row0 + group * 8, column0, n, acc[0]);
                    c.store_matrix(row0 + group * 8, column0 + 8, n, acc[1]);
                } else {
                    tile_c.store_matrix(group * 8, 0, 16, acc[0]);
                    tile_c.store_matrix(group * 8, 8, 16, acc[1]);
                    threadgroup::barrier();
                    let mut slot = thread;
                    while slot < 256 {
                        let (row, column) = (row0 + slot / 16, column0 + slot % 16);
                        if row < m && column < n {
                            c.store_unchecked(row * n + column, tile_c.load_unchecked(slot));
                        }
                        slot += 64;
                    }
                }
            }
        }
    };
}
gemm!(gemm_f16, false);
gemm!(gemm_nt_f16, true);
