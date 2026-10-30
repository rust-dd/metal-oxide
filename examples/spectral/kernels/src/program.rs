use metal_oxide_device::{Float, ReadBuffer, WriteBuffer, kernel};

// Device control flow uses scalar comparisons; borrowed range methods are unsupported.
#[allow(clippy::manual_range_contains)]
fn evaluate(program: ReadBuffer<f32>, len: u32, curve: f32, state: f32) -> f32 {
    let mut stack = [0.0; 8];
    let mut depth = 0_u32;
    let mut k = 0;
    while k < len {
        // SAFETY: callers provide len opcode/value pairs.
        let (code, value) = unsafe {
            (
                program.load_unchecked(2 * k) as u32,
                program.load_unchecked(2 * k + 1),
            )
        };
        if code < 3 && depth < 8 {
            stack[depth as usize] = if code == 0 {
                curve
            } else if code == 1 {
                state
            } else {
                value
            };
            depth += 1;
        } else if code >= 3 && code <= 9 && depth >= 2 && depth <= 8 {
            let a = stack[(depth - 2) as usize];
            let b = stack[(depth - 1) as usize];
            stack[(depth - 2) as usize] = match code {
                3 => a + b,
                4 => a - b,
                5 => a * b,
                6 => a / b,
                7 => a.powf(b),
                8 => {
                    if a > b {
                        a
                    } else {
                        b
                    }
                }
                _ => {
                    if a < b {
                        a
                    } else {
                        b
                    }
                }
            };
            depth -= 1;
        } else if code >= 10 && code <= 15 && depth >= 1 && depth <= 8 {
            let value = stack[(depth - 1) as usize];
            stack[(depth - 1) as usize] = match code {
                10 => -value,
                11 => value.sqrt(),
                12 => value.exp(),
                13 => value.ln(),
                14 => {
                    if value < 0.0 {
                        -value
                    } else {
                        value
                    }
                }
                _ => value.tanh(),
            };
        } else {
            return f32::NAN;
        }
        k += 1;
    }
    if depth == 1 { stack[0] } else { f32::NAN }
}

/// # Safety
/// out covers rows * steps values; curve covers steps; program covers len pairs.
/// steps is at most 512 and all buffer/index products fit u32.
#[kernel]
pub unsafe fn history_program(
    out: WriteBuffer<f32>,
    curve: ReadBuffer<f32>,
    program: ReadBuffer<f32>,
    len: u32,
    rows: u32,
    steps: u32,
    first_path: u32,
) {
    let path = crate::index();
    if path < rows && steps <= 512 {
        let mut history = [0.0; 512];
        let mut drift = [0.0; 176];
        let mut jump = [0.0; 176];
        let mut state = 0.5 + (first_path + path) as f32 * 0.001;
        let mut i = 0_u32;
        while i < steps && i < 512 {
            // SAFETY: i is within the supplied curve and current path.
            let ct = unsafe { curve.load_unchecked(i) };
            let value = evaluate(program, len, ct, state);
            history[i as usize] = value;
            let mut convolution = 0.0;
            let mut k = 0;
            while k <= i {
                let at = i - k;
                if at < 512 {
                    // SAFETY: k <= i < steps bounds the curve.
                    convolution += unsafe { curve.load_unchecked(k) } * history[at as usize];
                }
                k += 1;
            }
            let mut lifted = 0.0;
            let mut l = 0_usize;
            while l < 176 {
                lifted += (drift[l] + jump[l]) / (l + 1) as f32;
                drift[l] = 0.99 * drift[l] + 0.001 * value;
                jump[l] = 0.98 * (jump[l] + 0.002 * ct);
                l += 1;
            }
            state += 0.01 * (convolution + lifted);
            // SAFETY: each path owns a disjoint complete output row.
            unsafe { out.store_unchecked(path * steps + i, state) };
            i += 1;
        }
    }
}

/// # Safety
/// out covers rows * steps; events covers count time/size pairs; table has at
/// least two ascending entries. steps >= 2, and all buffer/index products fit u32.
#[kernel]
pub unsafe fn series_clock(
    out: WriteBuffer<f32>,
    events: ReadBuffer<f32>,
    table: ReadBuffer<f32>,
    count: u32,
    table_len: u32,
    rows: u32,
    steps: u32,
) {
    let path = crate::index();
    if path < rows && steps >= 2 && table_len >= 2 {
        let mut position = 1;
        let mut state = 0.0;
        let mut i = 1;
        // SAFETY: each active path owns steps output values.
        unsafe { out.store_unchecked(path * steps, state) };
        while i < steps {
            let time = i as f32 / (steps - 1) as f32;
            let mut j = 0;
            while j < count {
                // SAFETY: j indexes a supplied time/size pair.
                let (ratio, size) = unsafe {
                    (
                        events.load_unchecked(j * 2) * (steps - 1) as f32,
                        events.load_unchecked(j * 2 + 1),
                    )
                };
                let mut cell = ratio as u32;
                if (cell as f32) < ratio {
                    cell += 1;
                }
                if cell < 1 {
                    cell = 1;
                }
                if cell > steps - 1 {
                    cell = steps - 1;
                }
                if cell == i {
                    state += size;
                }
                j += 1;
            }
            // SAFETY: short circuit bounds table access; position starts at one.
            unsafe {
                while position < table_len && table.load_unchecked(position) < time {
                    position += 1;
                }
                let clock = if position >= table_len {
                    (table_len - 1) as f32
                } else {
                    let low = table.load_unchecked(position - 1);
                    let high = table.load_unchecked(position);
                    if high <= low {
                        position as f32
                    } else {
                        (position - 1) as f32 + (time - low) / (high - low)
                    }
                };
                out.store_unchecked(path * steps + i, state + clock);
            }
            i += 1;
        }
    }
}
