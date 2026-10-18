use fft::{cpu, half_spans, launch};
use metal_oxide::Dim3;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a Metal device"]
fn generated_fft_matches_cpu_and_msl() -> metal_oxide::Result<()> {
    super::run::verify()
}

#[test]
fn reference_transform_matches_a_direct_dft() {
    let input = (0..16)
        .map(|i| ((i * i) as f64, (i % 3) as f64))
        .collect::<Vec<_>>();
    let mut actual = input.clone();
    cpu::transform(&mut actual);
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
fn reference_transforms_each_row_on_its_own() {
    let (re, im) = ([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [0.0; 8]);
    let rows = cpu::reference(4, &re, &im);
    let impulse = [(1.0, 0.0); 4];
    let constant = [(4.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
    for (actual, expected) in rows.iter().zip(impulse.iter().chain(&constant)) {
        assert!((actual.0 - expected.0).abs() < 1e-12 && (actual.1 - expected.1).abs() < 1e-12);
    }
}

#[test]
fn stages_double_the_half_span_up_to_half_a_row() {
    assert_eq!(half_spans(2).collect::<Vec<_>>(), [1]);
    assert_eq!(half_spans(8).collect::<Vec<_>>(), [1, 2, 4]);
    assert_eq!(half_spans(4096).count(), 12);
}

#[test]
#[should_panic(expected = "12 values per row")]
fn stages_need_a_power_of_two() {
    half_spans(12).for_each(drop);
}

#[test]
fn a_launch_covers_every_butterfly_of_every_row() {
    assert_eq!(launch(2, 3).grid, Dim3::xy(1, 3));
    assert_eq!(launch(1024, 1).grid, Dim3::xy(2, 1));
    assert_eq!(launch(4096, 256).grid, Dim3::xy(8, 256));
}

#[test]
fn error_counts_tolerances_of_each_rows_rms() {
    // One stage allows 4e-6 in a zero row and 4e-5 in a row of rms 10.
    let expected = [(0.0, 0.0), (0.0, 0.0), (10.0, 0.0), (0.0, 10.0)];
    let error = |re: [f32; 4]| cpu::error(2, &expected, &re, &[0.0, 0.0, 0.0, 10.0]);
    let offset = 2.0_f32.powi(-15);
    let zero_row = error([2e-6, 0.0, 10.0, 0.0]) - f64::from(2e-6_f32) / 4e-6;
    let loud_row = error([0.0, 0.0, 10.0 + offset, 0.0]) - f64::from(offset) / 4e-5;
    assert!(
        zero_row.abs() < 1e-6 && loud_row.abs() < 1e-6,
        "{zero_row} {loud_row}"
    );
    assert_eq!(error([f32::NAN, 0.0, 10.0, 0.0]), f64::INFINITY);
}
