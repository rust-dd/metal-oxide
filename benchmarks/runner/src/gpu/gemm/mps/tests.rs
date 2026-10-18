use super::*;

#[test]
fn pad_rows_zero_fills_each_row_up_to_the_stride() {
    let tight = [1, 2, 3, 4, 5, 6];
    assert_eq!(pad_rows(&tight, 2, 5), [1, 2, 3, 0, 0, 4, 5, 6, 0, 0]);
    assert_eq!(pad_rows(&tight, 2, 3), tight);
    assert_eq!(pad_rows(&tight, 1, 8), [1, 2, 3, 4, 5, 6, 0, 0]);
}

#[test]
fn unpad_rows_skips_padding_whatever_it_holds_and_replaces_old_values() {
    let mut bytes = Vec::new();
    for row in [[1.0_f32, 2.0], [3.0, 4.0]] {
        for value in row.into_iter().chain([f32::NAN]) {
            bytes.extend(value.to_ne_bytes());
        }
    }
    let mut values = vec![9.0];
    unpad_rows(&bytes, 2, 12, &mut values);
    assert_eq!(values, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn padding_and_unpadding_round_trip_a_ragged_matrix() {
    let values = (0..15).map(|i| i as f32 * 0.25 - 1.0).collect::<Vec<_>>();
    let tight = values
        .iter()
        .flat_map(|x| x.to_ne_bytes())
        .collect::<Vec<_>>();
    let padded = pad_rows(&tight, 3, 32);
    assert_eq!(padded.len(), 96);
    let mut unpadded = Vec::new();
    unpad_rows(&padded, 5, 32, &mut unpadded);
    assert_eq!(unpadded, values);
}

#[test]
fn gpu_time_needs_two_valid_ordered_timestamps() {
    assert_eq!(gpu_ns(12.0, 12.25), Some(250_000_000));
    assert_eq!(gpu_ns(3.0, 3.0), Some(0));
    for (start, end) in [
        (0.0, 1.0),
        (f64::NAN, 1.0),
        (1.0, f64::INFINITY),
        (2.0, 1.0),
    ] {
        assert_eq!(gpu_ns(start, end), None, "{start} {end}");
    }
}

#[test]
fn mps_cases_repeat_like_the_others_but_have_no_threadgroup_size() {
    let case = serde_json::to_value(new_case("gemm-f16", [3, 4, 5], 8)).unwrap();
    assert_eq!(case["variant"], "mps");
    assert_eq!(case["shape"], serde_json::json!([3, 4, 5]));
    assert_eq!(case["repeat"], 8);
    assert!(case["block"].is_null());
}
