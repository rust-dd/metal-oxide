pub fn transform(values: &mut [(f64, f64)]) {
    let n = values.len();
    assert!(n.is_power_of_two());
    if n == 1 {
        return;
    }
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - n.trailing_zeros());
        if j > i {
            values.swap(i, j);
        }
    }
    let mut width = 2;
    while width <= n {
        for base in (0..n).step_by(width) {
            for pos in 0..width / 2 {
                let angle = -std::f64::consts::TAU * pos as f64 / width as f64;
                let (sin, cos) = angle.sin_cos();
                let a = values[base + pos];
                let b = values[base + pos + width / 2];
                let tr = b.0 * cos - b.1 * sin;
                let ti = b.0 * sin + b.1 * cos;
                values[base + pos] = (a.0 + tr, a.1 + ti);
                values[base + pos + width / 2] = (a.0 - tr, a.1 - ti);
            }
        }
        width *= 2;
    }
}

pub fn eigenvalues(n: usize, hurst: f64) -> Vec<f32> {
    let half = n / 2;
    let mut values = vec![(0.0, 0.0); n];
    values[0].0 = 1.0;
    for k in 1..=half {
        let kf = k as f64;
        let covariance = 0.5
            * ((kf + 1.0).powf(2.0 * hurst) - 2.0 * kf.powf(2.0 * hurst)
                + (kf - 1.0).powf(2.0 * hurst));
        values[k].0 = covariance;
        if k < half {
            values[n - k].0 = covariance;
        }
    }
    transform(&mut values);
    values
        .into_iter()
        .map(|(re, _)| (re / n as f64).max(0.0).sqrt() as f32)
        .collect()
}

pub fn reverse(n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| i.reverse_bits() >> (u32::BITS - n.trailing_zeros()))
        .collect()
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub fn normal_pair(cell: u64, seed: u32) -> (f32, f32) {
    let key = cell ^ (seed as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let normal = |hash: u64| {
        let u = (((hash as u32) >> 8) as f32 + 0.5) / 16777216.0;
        let v = (((hash >> 32) as u32 >> 8) as f32 + 0.5) / 16777216.0;
        (-2.0 * (u + 1e-10).ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    };
    (normal(mix(key)), normal(mix(key ^ 0xd1b5_4a32_d192_ed03)))
}

pub fn fgn(
    roots: &[f32],
    output_size: usize,
    rows: usize,
    first_row: u64,
    seed: u32,
    scale: f32,
) -> Vec<f32> {
    let n = roots.len();
    let parity = first_row as usize & 1;
    let base_cell = first_row.wrapping_sub(parity as u64).wrapping_mul(n as u64);
    let mut output = vec![0.0; rows * output_size];
    for batch in 0..(rows + parity).div_ceil(2) {
        let cell = base_cell.wrapping_add(batch as u64 * 2 * n as u64);
        let mut values = roots
            .iter()
            .enumerate()
            .map(|(i, eig)| {
                let z = normal_pair(cell.wrapping_add(i as u64), seed);
                ((z.0 * eig) as f64, (z.1 * eig) as f64)
            })
            .collect::<Vec<_>>();
        transform(&mut values);
        for h in 0..2 {
            let row = (2 * batch + h) as isize - parity as isize;
            if row < 0 || row as usize >= rows {
                continue;
            }
            for i in 0..output_size {
                let value = if h == 0 {
                    values[i + 1].0
                } else {
                    values[i + 1].1
                };
                output[row as usize * output_size + i] = value as f32 * scale;
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_fft_matches_direct_dft() {
        let input = (0..16)
            .map(|i| ((i * i) as f64, (i % 3) as f64))
            .collect::<Vec<_>>();
        let mut actual = input.clone();
        transform(&mut actual);
        for (k, value) in actual.iter().enumerate() {
            let mut expected = (0.0, 0.0);
            for (i, x) in input.iter().enumerate() {
                let (sin, cos) =
                    (-std::f64::consts::TAU * (i * k) as f64 / input.len() as f64).sin_cos();
                expected.0 += x.0 * cos - x.1 * sin;
                expected.1 += x.0 * sin + x.1 * cos;
            }
            assert!((value.0 - expected.0).abs() < 1e-10);
            assert!((value.1 - expected.1).abs() < 1e-10);
        }
    }

    #[test]
    fn white_noise_eigenvalues_and_bit_reversal_match_closed_forms() {
        assert_eq!(eigenvalues(16, 0.5), vec![0.25; 16]);
        assert_eq!(reverse(8), [0, 4, 2, 6, 1, 5, 3, 7]);
    }

    #[test]
    fn seeded_reference_is_invariant_to_odd_row_chunks() {
        let roots = eigenvalues(32, 0.7);
        let full = fgn(&roots, 13, 7, u32::MAX as u64, 123, 0.25);
        let mut chunks = fgn(&roots, 13, 3, u32::MAX as u64, 123, 0.25);
        chunks.extend(fgn(&roots, 13, 4, u32::MAX as u64 + 3, 123, 0.25));
        assert_eq!(full, chunks);
    }
}
