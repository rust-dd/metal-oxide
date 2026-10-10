use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};

/// # Safety
/// Buffers cover total cells in complete sheets; cells = rows * cols. lam covers
/// cells entries and reverse contains the length-cols bit-reversal permutation.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sheet_generate(
    real: WriteBuffer<f32>,
    imag: WriteBuffer<f32>,
    lam: ReadBuffer<f32>,
    reverse: ReadBuffer<u32>,
    cells: u32,
    cols: u32,
    seed: u32,
    first_cell: u64,
    total: u32,
) {
    let tid = crate::index();
    if tid < total && cells > 0 && cols > 0 {
        let sheet = tid / cells;
        let local = tid % cells;
        let row = local / cols;
        let col = local % cols;
        let pair = crate::random::normal_pair(first_cell.wrapping_add(tid as u64), seed);
        // SAFETY: each lane owns a distinct cell under the supplied permutation.
        unsafe {
            let scale = lam.load_unchecked(local);
            let destination = sheet * cells + row * cols + reverse.load_unchecked(col);
            real.store_unchecked(destination, pair.0 * scale);
            imag.store_unchecked(destination, pair.1 * scale);
        }
    }
}

/// # Safety
/// Inputs and outputs are disjoint, cover total cells and contain complete
/// rows-by-cols sheets. reverse is the length-rows bit-reversal permutation.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sheet_transpose(
    real: ReadBuffer<f32>,
    imag: ReadBuffer<f32>,
    out_real: WriteBuffer<f32>,
    out_imag: WriteBuffer<f32>,
    reverse: ReadBuffer<u32>,
    cells: u32,
    rows: u32,
    cols: u32,
    total: u32,
) {
    let tid = crate::index();
    if tid < total && cells > 0 && cols > 0 {
        let sheet = tid / cells;
        let local = tid % cells;
        let row = local / cols;
        let col = local % cols;
        // SAFETY: transpose and bit reversal map each source to a unique output cell.
        unsafe {
            let destination = sheet * cells + col * rows + reverse.load_unchecked(row);
            out_real.store_unchecked(destination, real.load_unchecked(tid));
            out_imag.store_unchecked(destination, imag.load_unchecked(tid));
        }
    }
}

#[derive(Clone, Copy)]
pub struct SheetReadout {
    pub cells: u32,
    pub rows: u32,
    pub m: u32,
    pub n: u32,
    pub radius: f32,
    pub correction: f32,
    pub seed: u32,
    pub correction_cell: u64,
}

/// # Safety
/// Inputs contain complete transposed sheets with rows entries per column.
/// Output covers total values in m-by-n crops contained in those sheets.
#[kernel]
pub unsafe fn sheet_extract(
    real: ReadBuffer<f32>,
    output: WriteBuffer<f32>,
    config: SheetReadout,
    total: u32,
) {
    let tid = crate::index();
    let per = config.m * config.n;
    if tid < total && per > 0 && config.n > 0 {
        let sheet = tid / per;
        let local = tid % per;
        let i = local / config.n;
        let j = local % config.n;
        let base = sheet * config.cells;
        let z = crate::random::normal_pair(
            config.correction_cell.wrapping_add(sheet as u64),
            config.seed,
        );
        let ty = config.radius * (i + 1) as f32 / config.m as f32;
        let tx = config.radius * (j + 1) as f32 / config.n as f32;
        // SAFETY: each lane owns one crop output and reads in-bounds transposed cells.
        unsafe {
            let value = real.load_unchecked(base + j * config.rows + i) - real.load_unchecked(base);
            output.store_unchecked(tid, value + config.correction * (ty * z.0 + tx * z.1));
        }
    }
}
