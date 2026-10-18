use metal_oxide::Dim3;
use stencil::{launch, reference, sweep, values};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a Metal device"]
fn generated_stencil_matches_cpu_and_msl() -> metal_oxide::Result<()> {
    super::run::verify()
}

#[test]
fn a_launch_has_a_block_for_each_tile_of_16_by_16_cells() {
    assert_eq!(launch(1, 1).grid, Dim3::xy(1, 1));
    assert_eq!(launch(16, 32).grid, Dim3::xy(1, 2));
    assert_eq!(launch(17, 33).grid, Dim3::xy(2, 3));
    assert_eq!(launch(2048, 2048).grid, Dim3::xy(128, 128));
}

/// The swept centre of a 3 x 3 grid with these left, right, upper and lower neighbours.
fn centre(l: f32, r: f32, u: f32, d: f32) -> f32 {
    let input = [9.0, u, 9.0, l, 9.0, r, 9.0, d, 9.0];
    let mut output = [0.0; 9];
    sweep(&input, &mut output, 3, 3);
    output[4]
}

#[test]
fn interior_cells_add_left_right_up_then_down() {
    // Every other order or grouping of these additions rounds to another sum.
    let tiny = 2.0_f32.powi(-25);
    let sum = centre(tiny, 1.5 * tiny, 1.0, 2.0 * tiny);
    assert_eq!(sum.to_bits(), (0.25 + 2.0 * tiny).to_bits());
}

#[test]
fn boundary_cells_copy_their_input() {
    let input = values(6 * 5);
    let mut output = vec![0.0; input.len()];
    sweep(&input, &mut output, 6, 5);
    for (i, (output, input)) in output.iter().zip(&input).enumerate() {
        let (x, y) = (i % 6, i / 6);
        if x == 0 || x == 5 || y == 0 || y == 4 {
            assert_eq!(output.to_bits(), input.to_bits(), "cell {i}");
        }
    }
    for (width, height) in [(1, 1), (2, 3), (3, 2), (1, 5)] {
        let input = values(width * height);
        let mut output = vec![0.0; input.len()];
        sweep(&input, &mut output, width as usize, height as usize);
        assert_eq!(output, input, "{width} x {height}");
    }
}

#[test]
#[should_panic(expected = "grids of 3 x 3")]
fn a_sweep_needs_grids_of_its_shape() {
    sweep(&[1.0; 9], &mut [0.0; 8], 3, 3);
}

#[test]
fn the_reference_alternates_between_two_grids() {
    let input = values(5 * 4);
    let mut once = vec![0.0; input.len()];
    let mut twice = vec![0.0; input.len()];
    sweep(&input, &mut once, 5, 4);
    sweep(&once, &mut twice, 5, 4);
    assert_eq!(reference(&input, 5, 4, 0), input);
    assert_eq!(reference(&input, 5, 4, 1), once);
    assert_eq!(reference(&input, 5, 4, 2), twice);
}

#[test]
fn values_are_positive_with_eight_exponents() {
    let values = values(1 << 12);
    assert!(values.iter().all(|value| (0.0625..16.0).contains(value)));
    let mut exponents = values
        .iter()
        .map(|value| value.to_bits() >> 23)
        .collect::<Vec<_>>();
    exponents.sort_unstable();
    exponents.dedup();
    assert_eq!(exponents, (123..131).collect::<Vec<_>>());
}
