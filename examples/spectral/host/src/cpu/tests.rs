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
