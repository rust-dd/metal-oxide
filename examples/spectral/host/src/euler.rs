use metal_oxide::{Argument, Device, Launch, LaunchConfig, Module, Operation, Pipeline};
use metal_oxide_artifact::MathMode;

use crate::{Result, kernels, validation::compare};

pub fn run(device: &Device, directory: &str, mode: MathMode) -> Result<()> {
    let kernels = kernels::load(device, directory)?;
    let source = include_str!("../euler.metal");
    let source = if mode == MathMode::Fast {
        source.replace("precise::", "fast::").replace(
            "#pragma STDC FP_CONTRACT OFF",
            "#pragma METAL fp math_mode(relaxed)\n#pragma METAL fp contract(fast)",
        )
    } else {
        source.into()
    };
    let native = Pipeline::new(
        device,
        &Module::from_source(device, &source)?,
        "sample_jumps",
    )?;
    for (shape, intensity) in [(-1.0, 0.0), (0.2, 0.75), (2.5, 8.0), (0.5, 100.0)] {
        let (rows, steps, first, seed) = (7, 129, u32::MAX - 3, 123);
        let mut generated = device.buffer_zeroed::<f32>((rows * steps * 4) as usize)?;
        let mut expected = device.buffer_zeroed::<f32>(generated.len())?;
        // SAFETY: both kernels own four outputs per active path/step cell.
        unsafe {
            kernels
                .sample_jumps(
                    LaunchConfig::<64>::for_elements(rows * steps)?,
                    &mut generated,
                    rows,
                    steps,
                    first,
                    seed,
                    shape,
                    intensity,
                )
                .wait()?;
            Launch::new(
                Ok(&native),
                LaunchConfig::<64>::for_elements(rows * steps)?,
                (&mut expected, rows, steps, first, seed, shape, intensity),
                |(expected, rows, steps, first, seed, shape, intensity)| {
                    Ok([
                        Argument::write(expected),
                        Argument::value(*rows)?,
                        Argument::value(*steps)?,
                        Argument::value(*first)?,
                        Argument::value(*seed)?,
                        Argument::value(*shape)?,
                        Argument::value(*intensity)?,
                    ])
                },
            )
            .wait()?;
        }
        compare(
            generated.read().wait(),
            expected.read().wait(),
            1.0,
            "Euler native draws",
        );
        for (cell, values) in generated
            .read()
            .wait()
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
        {
            let path = first.wrapping_add(cell as u32 / steps);
            let key = hash(
                path.wrapping_mul(2_654_435_761)
                    ^ (cell as u32 % steps)
                        .wrapping_add(2_246_822_519)
                        .wrapping_mul(2_246_822_519),
            );
            let mut product = 1.0;
            let count = (0..64_u32)
                .take_while(|&j| {
                    let word = hash(
                        (key ^ 2_166_136_261_u32.wrapping_add(j.wrapping_mul(16_777_619)))
                            ^ seed.wrapping_mul(374_761_393),
                    );
                    // Metal flushes subnormal arithmetic at very high intensities.
                    product = flush(product * (word as f32 * 2.328_306_4e-10));
                    product > flush((-intensity).exp())
                })
                .count();
            // Fast reassociation has no CPU-equivalent subnormal product order.
            // The extreme-intensity case still requires exact native integral counts.
            if mode == MathMode::Precise || intensity < 80.0 {
                assert_eq!(values[0], count as f32, "Euler CPU count");
            }
        }
        let mut chunked = Vec::new();
        for path in 0..rows {
            let mut chunk = device.buffer_zeroed::<f32>((steps * 4) as usize)?;
            // SAFETY: one complete path, with the same global key as the full launch.
            unsafe {
                kernels.sample_jumps(
                    LaunchConfig::<32>::for_elements(steps)?,
                    &mut chunk,
                    1,
                    steps,
                    first.wrapping_add(path),
                    seed,
                    shape,
                    intensity,
                )
            }
            .wait()?;
            chunked.extend_from_slice(chunk.read().wait());
        }
        assert_eq!(
            generated.read().wait(),
            chunked,
            "Euler draw chunk invariance"
        );
    }
    history(device, &kernels)?;
    series(device, &kernels)?;
    println!(
        "Euler: count/gamma/jump parity, program/history/lift/series/table and chunk invariance"
    );
    Ok(())
}

fn hash(mut word: u32) -> u32 {
    word ^= word >> 16;
    word = word.wrapping_mul(2_246_822_519);
    word ^= word >> 13;
    word = word.wrapping_mul(3_266_489_917);
    word ^ (word >> 16)
}

fn flush(value: f32) -> f32 {
    if value.is_subnormal() { 0.0 } else { value }
}

fn evaluate(program: &[(u32, f32)], curve: f32, state: f32) -> f32 {
    let mut stack = Vec::new();
    for &(code, value) in program {
        if code <= 2 {
            stack.push(match code {
                0 => curve,
                1 => state,
                _ => value,
            });
        } else if code <= 9 {
            let b = stack.pop().unwrap();
            let a = stack.pop().unwrap();
            stack.push(match code {
                3 => a + b,
                4 => a - b,
                5 => a * b,
                6 => a / b,
                7 => a.powf(b),
                8 => a.max(b),
                _ => a.min(b),
            });
        } else {
            let a = stack.pop().unwrap();
            stack.push(match code {
                10 => -a,
                11 => a.sqrt(),
                12 => a.exp(),
                13 => a.ln(),
                14 => a.abs(),
                _ => a.tanh(),
            });
        }
    }
    assert_eq!(stack.len(), 1);
    stack[0]
}

fn history(device: &Device, kernels: &kernels::Kernels) -> Result<()> {
    let program = [
        (0, 0.0),
        (1, 0.0),
        (3, 0.0),
        (2, 0.5),
        (5, 0.0),
        (2, 2.0),
        (7, 0.0),
        (11, 0.0),
        (12, 0.0),
        (13, 0.0),
        (10, 0.0),
        (14, 0.0),
        (15, 0.0),
        (2, 0.01),
        (8, 0.0),
        (2, 0.9),
        (9, 0.0),
        (2, 0.1),
        (4, 0.0),
        (2, 2.0),
        (6, 0.0),
    ];
    let instructions = device.buffer_from_slice(
        &program
            .iter()
            .flat_map(|&(code, value)| [code as f32, value])
            .collect::<Vec<_>>(),
    )?;
    for steps in [1, 17, 129, 512_u32] {
        let rows = 3;
        let first = 7;
        let curve = (0..steps).map(|i| 0.1 / (i + 1) as f32).collect::<Vec<_>>();
        let expected = (first..first + rows)
            .flat_map(|path| {
                let mut state = 0.5 + path as f32 * 0.001;
                let mut history = Vec::new();
                let mut drift = vec![0.0; 176];
                let mut jump = vec![0.0; 176];
                let mut row = Vec::new();
                for &ct in &curve {
                    let value = evaluate(&program, ct, state);
                    history.push(value);
                    let convolution = curve
                        .iter()
                        .zip(history.iter().rev())
                        .map(|(c, h)| c * h)
                        .sum::<f32>();
                    let lifted = drift
                        .iter()
                        .zip(&jump)
                        .enumerate()
                        .map(|(l, (d, j))| (d + j) / (l + 1) as f32)
                        .sum::<f32>();
                    for (d, j) in drift.iter_mut().zip(&mut jump) {
                        *d = 0.99 * *d + 0.001 * value;
                        *j = 0.98 * (*j + 0.002 * ct);
                    }
                    state += 0.01 * (convolution + lifted);
                    row.push(state);
                }
                row
            })
            .collect::<Vec<_>>();
        let curve = device.buffer_from_slice(&curve)?;
        let mut actual = device.buffer_zeroed::<f32>(expected.len())?;
        // SAFETY: the program has a valid stack of depth <= 8; all rows and curves
        // cover steps <= 512. The history and lift arrays are private per lane.
        unsafe {
            kernels.history_program(
                LaunchConfig::<32>::for_elements(rows)?,
                &mut actual,
                &curve,
                &instructions,
                program.len() as u32,
                rows,
                steps,
                first,
            )
        }
        .wait()?;
        compare(
            actual.read().wait(),
            &expected,
            1.0,
            "program/history/lift CPU",
        );
    }
    Ok(())
}

fn series(device: &Device, kernels: &kernels::Kernels) -> Result<()> {
    let events = [0.0, 0.5, 0.125, 0.25, 0.3, -0.125, 1.0, 0.75];
    let table = [0.0, 0.25, 0.25, 0.4, 0.6];
    let (rows, steps) = (3, 33);
    let mut row = vec![0.0];
    let mut accumulated = 0.0;
    for i in 1..steps {
        for pair in events.as_chunks::<2>().0 {
            let cell = (pair[0] * (steps - 1) as f32)
                .ceil()
                .max(1.0)
                .min((steps - 1) as f32) as u32;
            if cell == i {
                accumulated += pair[1];
            }
        }
        let time = i as f32 / (steps - 1) as f32;
        let high = table[1..].iter().position(|&x| x >= time).map(|i| i + 1);
        let clock = high.map_or((table.len() - 1) as f32, |hi| {
            let lo = hi - 1;
            if table[hi] <= table[lo] {
                hi as f32
            } else {
                lo as f32 + (time - table[lo]) / (table[hi] - table[lo])
            }
        });
        row.push(accumulated + clock);
    }
    let mut output = device.buffer_zeroed::<f32>((rows * steps) as usize)?;
    let events_buffer = device.buffer_from_slice(&events)?;
    let table_buffer = device.buffer_from_slice(&table)?;
    // SAFETY: input pairs and monotone table cover their counts; every row is complete.
    unsafe {
        kernels.series_clock(
            LaunchConfig::<32>::for_elements(rows)?,
            &mut output,
            &events_buffer,
            &table_buffer,
            events.len() as u32 / 2,
            table.len() as u32,
            rows,
            steps,
        )
    }
    .wait()?;
    compare(
        output.read().wait(),
        &row.repeat(rows as usize),
        1.0,
        "series/table CPU",
    );
    Ok(())
}
