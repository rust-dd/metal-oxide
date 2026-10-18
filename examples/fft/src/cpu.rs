//! The f64 reference transform and the tolerance that the kernel's results must meet.

/// The in-place forward transform of `values`, a power-of-two count, in f64: the spectral
/// example's iterative radix-2 FFT, with each twiddle computed once per stage.
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
        for pos in 0..width / 2 {
            let angle = -std::f64::consts::TAU * pos as f64 / width as f64;
            let (sin, cos) = angle.sin_cos();
            for base in (0..n).step_by(width) {
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

/// The transform of every row of `n` values, a power of two, in `re` and `im`.
pub fn reference(n: usize, re: &[f32], im: &[f32]) -> Vec<(f64, f64)> {
    assert!(
        re.len() == im.len() && re.len().is_multiple_of(n),
        "rows of {n}"
    );
    let mut values = re
        .iter()
        .zip(im)
        .map(|(&re, &im)| (f64::from(re), f64::from(im)))
        .collect::<Vec<_>>();
    for row in values.chunks_mut(n) {
        transform(row);
    }
    values
}

/// The largest distance of a result from `expected`, in units of its row's tolerance
/// `4e-6 * log2(n) * max(1, rms of the row's expected values)`; non-finite results are infinite.
pub fn error(n: usize, expected: &[(f64, f64)], re: &[f32], im: &[f32]) -> f64 {
    assert!(n >= 2 && n.is_power_of_two(), "{n} values per row");
    assert!(
        expected.len().is_multiple_of(n)
            && re.len() == expected.len()
            && im.len() == expected.len()
    );
    let stages = f64::from(n.trailing_zeros());
    let rows = expected.chunks(n).zip(re.chunks(n).zip(im.chunks(n)));
    let mut worst = 0.0_f64;
    for (expected, (re, im)) in rows {
        let power = expected
            .iter()
            .map(|(re, im)| re * re + im * im)
            .sum::<f64>();
        let tolerance = 4e-6 * stages * (power / n as f64).sqrt().max(1.0);
        for (&(want_re, want_im), (&re, &im)) in expected.iter().zip(re.iter().zip(im)) {
            let (dr, di) = (f64::from(re) - want_re, f64::from(im) - want_im);
            let distance = (dr * dr + di * di).sqrt();
            worst = worst.max(if distance.is_finite() {
                distance / tolerance
            } else {
                f64::INFINITY
            });
        }
    }
    worst
}
