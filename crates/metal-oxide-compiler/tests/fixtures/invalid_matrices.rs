#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel, simdgroup::{self, Matrix}};
fn store_through(output: WriteBuffer<f32>, matrix: &Matrix<f32>) {
    unsafe { output.store_matrix(0, 0, 8, *matrix) }
}
#[cfg(any(lane_branch, lane_operand, lane_value, lane_matrix, by_reference))]
#[kernel(block = (64, 1, 1))]
pub unsafe fn invalid(input: ReadBuffer<f32>, output: WriteBuffer<f32>) {
    let lane = simdgroup::lane_id();
    #[cfg(lane_branch)] let matrix = if lane < 16 { Matrix::splat(1.0) } else { Matrix::splat(0.0) };
    #[cfg(lane_operand)] let matrix = unsafe { input.load_matrix(lane, 0, 8) };
    #[cfg(lane_value)] let matrix = Matrix::splat(lane as f32);
    #[cfg(lane_matrix)] let (ones, zeros) = (Matrix::splat(1.0), Matrix::splat(0.0));
    #[cfg(lane_matrix)] let matrix = if lane < 16 { ones } else { zeros };
    #[cfg(by_reference)] let matrix = Matrix::splat(1.0);
    #[cfg(by_reference)] store_through(output, &matrix);
    #[cfg(not(by_reference))] unsafe { output.store_matrix(0, 0, 8, matrix) };
    let _ = input;
}
#[cfg(no_block)]
#[kernel]
pub unsafe fn no_block(output: WriteBuffer<f32>) {
    unsafe { output.store_matrix(0, 0, 8, Matrix::splat(1.0)) };
}
#[cfg(odd_block)]
#[kernel(block = (48, 1, 1))]
pub unsafe fn odd_block(output: WriteBuffer<f32>) {
    unsafe { output.store_matrix(0, 0, 8, Matrix::splat(1.0)) };
}
#[cfg(parameter)]
#[kernel(block = (64, 1, 1))]
pub unsafe fn parameter(output: WriteBuffer<f32>, matrix: Matrix<f32>) {
    unsafe { output.store_matrix(0, 0, 8, matrix) };
}
#[cfg(element)]
#[kernel(block = (64, 1, 1))]
pub unsafe fn element(input: ReadBuffer<Matrix<f32>>) {
    let _ = input;
}
