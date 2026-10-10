use metal_oxide::{Argument, Device, DynamicLaunchConfig, Module, Pipeline};
use metal_oxide_artifact::MathMode;

use crate::{Result, cpu, kernels::SheetReadout, validation::compare};

pub fn run(device: &Device, directory: &str, mode: MathMode) -> Result<()> {
    let generated = Module::from_artifact(device, directory)?;
    let source = format!(
        "{}\n{}",
        include_str!("../common.metal"),
        include_str!("../sheet.metal")
    );
    let source = if mode == MathMode::Fast {
        let mut source = format!(
            "#pragma METAL fp math_mode(relaxed)\n#pragma METAL fp contract(fast)\n{source}"
        );
        for function in ["sin", "cos", "sqrt", "log"] {
            source = source.replace(&format!("{function}("), &format!("fast::{function}("));
        }
        source
    } else {
        format!("#pragma STDC FP_CONTRACT OFF\n{source}")
    };
    let native = Module::from_source(device, &source)?;
    for (rows, cols, sheets) in [(2, 2, 1), (8, 16, 3), (32, 8, 2)] {
        let cells = rows * cols;
        let config = SheetReadout {
            cells,
            rows,
            m: rows / 2 + 1,
            n: cols / 2 + 1,
            radius: 1.5,
            correction: 0.125,
            seed: 123,
            correction_cell: u32::MAX as u64 + 999,
        };
        let lam = (0..cells)
            .map(|i| ((i % 13 + 1) as f32).sqrt() / cells as f32)
            .collect::<Vec<_>>();
        let first_cell = u32::MAX as u64 + 17;
        let want = reference(&lam, cols, sheets, first_cell, config);
        let rust = execute(device, &generated, true, &lam, sheets, first_cell, config)?;
        let metal = execute(device, &native, false, &lam, sheets, first_cell, config)?;
        compare(&rust, &want, 1.0, "sheet CPU");
        compare(&rust, &metal, 1.0, "sheet native");
        let mut chunked = Vec::new();
        for sheet in 0..sheets {
            let chunk = SheetReadout {
                correction_cell: config.correction_cell + sheet as u64,
                ..config
            };
            chunked.extend(execute(
                device,
                &generated,
                true,
                &lam,
                1,
                first_cell + sheet as u64 * cells as u64,
                chunk,
            )?);
        }
        assert_eq!(rust, chunked, "sheet chunk invariance {rows}x{cols}");
    }
    Ok(())
}

fn execute(
    device: &Device,
    module: &Module,
    generated: bool,
    lam: &[f32],
    sheets: u32,
    first: u64,
    config: SheetReadout,
) -> Result<Vec<f32>> {
    let cols = config.cells / config.rows;
    let pipelines = [
        "sheet_generate",
        "fft_butterfly",
        "sheet_transpose",
        "sheet_extract",
    ]
    .into_iter()
    .map(|name| Pipeline::new(device, module, name))
    .collect::<metal_oxide::Result<Vec<_>>>()?;
    let total = config.cells * sheets;
    let mut real = device.buffer_zeroed::<f32>(total as usize)?;
    let mut imag = device.buffer_zeroed::<f32>(total as usize)?;
    let mut transposed_real = device.buffer_zeroed::<f32>(total as usize)?;
    let mut transposed_imag = device.buffer_zeroed::<f32>(total as usize)?;
    let mut output = device.buffer_zeroed::<f32>((config.m * config.n * sheets) as usize)?;
    let roots = device.buffer_from_slice(lam)?;
    let reverse_cols = device.buffer_from_slice(&cpu::reverse(cols as usize))?;
    let reverse_rows = device.buffer_from_slice(&cpu::reverse(config.rows as usize))?;
    let block = if generated { 256 } else { 1 };
    // SAFETY: both dimensions are powers of two. The permutation stages own unique
    // cells, ordered FFT passes own disjoint pairs, and the crop stays within each sheet.
    // Native kernels have no tail guard, so their one-thread blocks cover exact counts.
    unsafe {
        device
            .submit(|batch| {
                let mut arguments = vec![
                    Argument::write(&mut real),
                    Argument::write(&mut imag),
                    Argument::read(&roots),
                    Argument::read(&reverse_cols),
                    Argument::value(config.cells)?,
                    Argument::value(cols)?,
                    Argument::value(config.seed)?,
                    Argument::value(first)?,
                ];
                if generated {
                    arguments.push(Argument::value(total)?);
                }
                batch.launch(
                    &pipelines[0],
                    DynamicLaunchConfig::for_elements(total, block)?,
                    &arguments,
                )?;
                drop(arguments);
                for stage in 0..cols.trailing_zeros() {
                    let mut arguments = vec![
                        Argument::read_write(&mut real),
                        Argument::read_write(&mut imag),
                        Argument::value(cols)?,
                        Argument::value(1_u32 << stage)?,
                    ];
                    if generated {
                        arguments.push(Argument::value(total / 2)?);
                    }
                    batch.launch(
                        &pipelines[1],
                        DynamicLaunchConfig::for_elements(total / 2, block)?,
                        &arguments,
                    )?;
                }
                let mut arguments = vec![
                    Argument::read(&real),
                    Argument::read(&imag),
                    Argument::write(&mut transposed_real),
                    Argument::write(&mut transposed_imag),
                    Argument::read(&reverse_rows),
                    Argument::value(config.cells)?,
                    Argument::value(config.rows)?,
                    Argument::value(cols)?,
                ];
                if generated {
                    arguments.push(Argument::value(total)?);
                }
                batch.launch(
                    &pipelines[2],
                    DynamicLaunchConfig::for_elements(total, block)?,
                    &arguments,
                )?;
                drop(arguments);
                for stage in 0..config.rows.trailing_zeros() {
                    let mut arguments = vec![
                        Argument::read_write(&mut transposed_real),
                        Argument::read_write(&mut transposed_imag),
                        Argument::value(config.rows)?,
                        Argument::value(1_u32 << stage)?,
                    ];
                    if generated {
                        arguments.push(Argument::value(total / 2)?);
                    }
                    batch.launch(
                        &pipelines[1],
                        DynamicLaunchConfig::for_elements(total / 2, block)?,
                        &arguments,
                    )?;
                }
                let out_len = config.m * config.n * sheets;
                let mut arguments = vec![
                    Argument::read(&transposed_real),
                    Argument::write(&mut output),
                ];
                if generated {
                    arguments.extend([Argument::value(config)?, Argument::value(out_len)?]);
                } else {
                    arguments.extend([
                        Argument::value(config.cells)?,
                        Argument::value(config.rows)?,
                        Argument::value(config.m)?,
                        Argument::value(config.n)?,
                        Argument::value(config.radius)?,
                        Argument::value(config.correction)?,
                        Argument::value(config.seed)?,
                        Argument::value(config.correction_cell)?,
                    ]);
                }
                batch.launch(
                    &pipelines[3],
                    DynamicLaunchConfig::for_elements(out_len, block)?,
                    &arguments,
                )
            })?
            .wait()?;
    }
    Ok(output.as_slice().to_vec())
}

fn reference(lam: &[f32], cols: u32, sheets: u32, first: u64, config: SheetReadout) -> Vec<f32> {
    let mut output = Vec::new();
    for sheet in 0..sheets {
        let mut values = lam
            .iter()
            .enumerate()
            .map(|(i, &weight)| {
                let z = cpu::normal_pair(
                    first + sheet as u64 * config.cells as u64 + i as u64,
                    config.seed,
                );
                ((z.0 * weight) as f64, (z.1 * weight) as f64)
            })
            .collect::<Vec<_>>();
        for row in values.chunks_mut(cols as usize) {
            cpu::transform(row);
        }
        for col in 0..cols as usize {
            let mut column = (0..config.rows as usize)
                .map(|row| values[row * cols as usize + col])
                .collect::<Vec<_>>();
            cpu::transform(&mut column);
            for (row, value) in column.into_iter().enumerate() {
                values[row * cols as usize + col] = value;
            }
        }
        let z = cpu::normal_pair(config.correction_cell + sheet as u64, config.seed);
        for i in 0..config.m {
            for j in 0..config.n {
                let value = (values[(i * cols + j) as usize].0 - values[0].0) as f32;
                let ty = config.radius * (i + 1) as f32 / config.m as f32;
                let tx = config.radius * (j + 1) as f32 / config.n as f32;
                output.push(value + config.correction * (ty * z.0 + tx * z.1));
            }
        }
    }
    output
}
