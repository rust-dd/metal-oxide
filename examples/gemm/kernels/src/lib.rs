#![no_std]

use metal_oxide_device::{
    F16, ReadBuffer, ThreadgroupBuffer, WriteBuffer, block_idx, kernel,
    simdgroup::{self, Matrix, MatrixElement},
    thread_idx, threadgroup,
};

/// An input element type, with the zero that pads partial tiles.
pub trait Input: MatrixElement {
    const ZERO: Self;
}

impl Input for f32 {
    const ZERO: Self = 0.0;
}

impl Input for F16 {
    const ZERO: Self = F16::from_bits(0);
}

/// A SIMD group's 32x32 part of a block tile, as 4x4 fragments of 8x8.
type Fragments = [[Matrix<f32>; 4]; 4];

/// The 64x16 slice of A or the 16x64 slice of B that a block stages for one K step.
type Staged<T> = ThreadgroupBuffer<T, 1024>;

/// Loads the four fragments of `$source` down from (`$row`, `$column`).
macro_rules! load_down {
    ($source:expr, $row:expr, $column:expr, $stride:expr) => {
        [
            $source.load_matrix($row, $column, $stride),
            $source.load_matrix($row + 8, $column, $stride),
            $source.load_matrix($row + 16, $column, $stride),
            $source.load_matrix($row + 24, $column, $stride),
        ]
    };
}

/// Loads the four fragments of `$source` across from (`$row`, `$column`).
macro_rules! load_across {
    ($source:expr, $row:expr, $column:expr, $stride:expr) => {
        [
            $source.load_matrix($row, $column, $stride),
            $source.load_matrix($row, $column + 8, $stride),
            $source.load_matrix($row, $column + 16, $stride),
            $source.load_matrix($row, $column + 24, $stride),
        ]
    };
}

/// `$acc` plus the products of the A fragments `$a` (down) and the B fragments `$b` (across).
macro_rules! multiply_slice {
    ($acc:expr, $a:expr, $b:expr) => {
        [
            multiply_row($acc[0], $a[0], $b),
            multiply_row($acc[1], $a[1], $b),
            multiply_row($acc[2], $a[2], $b),
            multiply_row($acc[3], $a[3], $b),
        ]
    };
}

/// Stores a SIMD group's fragments, the first at (`$row`, `$column`) of `$target`; literal
/// indices keep the fragments in registers.
macro_rules! store_fragments {
    (@pairs $target:expr, $acc:expr, $row:expr, $column:expr, $stride:expr;
        $($i:literal $j:literal)*) => {
        $($target.store_matrix($row + 8 * $i, $column + 8 * $j, $stride, $acc[$i][$j]);)*
    };
    ($target:expr, $acc:expr, $row:expr, $column:expr, $stride:expr) => {
        store_fragments!(@pairs $target, $acc, $row, $column, $stride;
            0 0 0 1 0 2 0 3 1 0 1 1 1 2 1 3 2 0 2 1 2 2 2 3 3 0 3 1 3 2 3 3)
    };
}

/// # Safety
/// Distinct row-major A, B and C cover m*k, k*n and m*n elements; these sizes and m, n and k
/// rounded up to 64 fit u32, and the grid is (ceil(n / 64), ceil(m / 64), 1) blocks.
#[kernel(instances(gemm_f32(f32), gemm_f16(F16)), block = (128, 1, 1))]
pub unsafe fn gemm<T: Input>(
    a: ReadBuffer<T>,
    b: ReadBuffer<T>,
    c: WriteBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) {
    let tile_a = threadgroup::shared::<T, 1024>();
    let tile_b = threadgroup::shared::<T, 1024>();
    let tile_c = threadgroup::shared::<f32, 1024>();
    let thread = thread_idx().x;
    let (row0, column0) = (block_idx().y * 64, block_idx().x * 64);
    // Four SIMD groups split the block tile 2x2.
    let group = simdgroup::group_id();
    let (row, column) = (group / 2 * 32, group % 2 * 32);
    let mut acc = [[Matrix::splat(0.0_f32); 4]; 4];
    let full = row0 + 64 <= m && column0 + 64 <= n;
    // SAFETY: the caller's contract bounds every global access and the block's tile of C is its
    // own; barriers that every thread reaches order the threadgroup accesses.
    unsafe {
        // On M4 Max (Apple9) device loads beat staging; edge blocks, or all if k % 16 > 0, stage.
        // `&`, not `&&`: the structured MSL emitter rejects the short-circuit form of this branch.
        if full & k.is_multiple_of(16) {
            acc = direct(acc, a, b, (row0 + row, column0 + column), (n, k));
        } else {
            let mut base = 0;
            while base < k {
                stage(a, b, tile_a, tile_b, (row0, column0, base), (m, n, k));
                threadgroup::barrier();
                acc = multiply(acc, tile_a, tile_b, (row, column));
                threadgroup::barrier();
                base += 16;
            }
        }
        if full {
            store_fragments!(c, acc, row0 + row, column0 + column, n);
            return;
        }
        // The groups take turns through one 32x32 tile: Metal's shader validation doubles static
        // threadgroup memory, so a 64x64 tile would exceed 32 KiB.
        for turn in 0..4 {
            if group == turn {
                store_fragments!(tile_c, acc, 0, 0, 32);
            }
            threadgroup::barrier();
            let (top, left) = (row0 + turn / 2 * 32, column0 + turn % 2 * 32);
            for step in 0..8 {
                let slot = thread + 128 * step;
                let (c_row, c_column) = (top + slot / 32, left + slot % 32);
                if c_row < m && c_column < n {
                    c.store_unchecked(c_row * n + c_column, tile_c.load_unchecked(slot));
                }
            }
            threadgroup::barrier();
        }
    }
}

/// Adds A * B to the fragments of the SIMD group whose 32x32 part of C starts at (`top`, `left`),
/// two k slices per step straight from device memory; that part lies inside C and 16 divides k.
unsafe fn direct<T: Input>(
    mut acc: Fragments,
    a: ReadBuffer<T>,
    b: ReadBuffer<T>,
    (top, left): (u32, u32),
    (n, k): (u32, u32),
) -> Fragments {
    let mut inner = 0;
    while inner < k {
        // SAFETY: rows top..top + 32 of A, columns left..left + 32 of B and depth inner + 16 exist.
        let (a0, b0, a1, b1) = unsafe {
            (
                load_down!(a, top, inner, k),
                load_across!(b, inner, left, n),
                load_down!(a, top, inner + 8, k),
                load_across!(b, inner + 8, left, n),
            )
        };
        acc = multiply_slice!(acc, a0, b0);
        acc = multiply_slice!(acc, a1, b1);
        inner += 16;
    }
    acc
}

/// Stages 8 elements of the A and B slices at depth `base` per thread, zero outside the matrices,
/// under the kernel's contract; barriers must separate it from every read of the tiles.
unsafe fn stage<T: Input>(
    a: ReadBuffer<T>,
    b: ReadBuffer<T>,
    tile_a: Staged<T>,
    tile_b: Staged<T>,
    (row0, column0, base): (u32, u32, u32),
    (m, n, k): (u32, u32, u32),
) {
    let thread = thread_idx().x;
    // SAFETY: global indices are guarded by the matrix sizes; slots stay below 1024.
    unsafe {
        for step in 0..8 {
            let slot = thread + 128 * step;
            let (a_row, a_inner) = (row0 + slot / 16, base + slot % 16);
            let x = if a_row < m && a_inner < k {
                a.load_unchecked(a_row * k + a_inner)
            } else {
                T::ZERO
            };
            tile_a.store_unchecked(slot, x);
            let (b_inner, b_column) = (base + slot / 64, column0 + slot % 64);
            let y = if b_inner < k && b_column < n {
                b.load_unchecked(b_inner * n + b_column)
            } else {
                T::ZERO
            };
            tile_b.store_unchecked(slot, y);
        }
    }
}

/// Adds the staged K step to the fragments of the SIMD group at (`row`, `column`) of the block
/// tile, both at most 32; the tiles must be staged, and no thread may write them meanwhile.
unsafe fn multiply<T: Input>(
    mut acc: Fragments,
    tile_a: Staged<T>,
    tile_b: Staged<T>,
    (row, column): (u32, u32),
) -> Fragments {
    for slice in 0..2 {
        let inner = 8 * slice;
        // SAFETY: the tiles are staged, and row and column of at most 32 keep each load inside.
        let (a, b) = unsafe {
            (
                load_down!(tile_a, row, inner, 16),
                load_across!(tile_b, inner, column, 64),
            )
        };
        acc = multiply_slice!(acc, a, b);
    }
    acc
}

fn multiply_row<T: Input>(
    acc: [Matrix<f32>; 4],
    a: Matrix<T>,
    b: [Matrix<T>; 4],
) -> [Matrix<f32>; 4] {
    [
        Matrix::multiply_accumulate(a, b[0], acc[0]),
        Matrix::multiply_accumulate(a, b[1], acc[1]),
        Matrix::multiply_accumulate(a, b[2], acc[2]),
        Matrix::multiply_accumulate(a, b[3], acc[3]),
    ]
}
