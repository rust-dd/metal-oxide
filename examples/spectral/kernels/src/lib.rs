#![no_std]

mod euler;
mod fft;
mod program;
mod random;
mod sampling;
mod sheet;
mod tile;

pub use euler::integrate;
pub use fft::{Readout, fft_butterfly, fft_butterfly_extract, fft_radix4, fft_radix4_extract};
pub use program::{history_program, series_clock};
pub use sampling::sample_jumps;
pub use sheet::{SheetReadout, sheet_extract, sheet_generate, sheet_transpose};
pub use tile::gen_tile_fft;

use metal_oxide_device::{block_dim, block_idx, thread_idx};

fn index() -> u32 {
    block_idx().x * block_dim().x + thread_idx().x
}
