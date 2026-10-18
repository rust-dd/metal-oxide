use super::*;

#[test]
fn product_reproduces_a_hand_computed_2x2x2_case() {
    let c = product(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], [2, 2, 2]);
    assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn product_keeps_the_axes_of_a_rectangular_case_apart() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0];
    assert_eq!(
        product(&a, &b, [2, 3, 4]),
        [12.0, 5.0, 12.0, 28.0, 13.0, 36.0]
    );
}

#[test]
fn problem_pairs_the_closed_form_inputs_with_their_product() {
    let problem = Problem::new([2, 2, 2]);
    assert_eq!(problem.a, [-1.0, -0.875, -0.75, -0.625]);
    assert_eq!(problem.b, [-1.5, -1.25, -1.0, -0.75]);
    assert_eq!(problem.expected, [2.375, 1.90625, 1.75, 1.40625]);
}

#[test]
fn check_accepts_only_bitwise_equal_results() {
    let expected = [1.0, 0.0, 3.0, 4.0];
    check("case", &expected, &expected, 2).unwrap();
    let message = |actual: &[f32]| check("case", actual, &expected, 2).unwrap_err().to_string();
    assert_eq!(
        message(&[1.0, 0.0, 3.0, 5.0]),
        "case: C[1][1] is 5 instead of 4"
    );
    assert!(message(&[1.0, -0.0, 3.0, 4.0]).contains("C[0][1]"));
    assert!(message(&[1.0, 0.0, f32::NAN, 4.0]).contains("C[1][0]"));
    assert!(message(&[1.0, 0.0, 3.0]).contains("3 elements"));
}

#[test]
fn gemm_cases_launch_128_threads_and_report_no_cpu_reference() {
    let sample = |cpu_reference_ns| Sample {
        cpu_reference_ns,
        submit_ns: 100,
        ..Sample::default()
    };
    let mut case = Case::new("gemm-f16", "msl", vec![2, 2, 2]);
    case.samples.push(sample(None));
    let case = case.finish().unwrap();
    assert_eq!(case.block, [128, 1, 1]);
    assert!(!case.summary_ns.contains_key("cpu_reference_ns"));
    assert!(serde_json::to_value(&case).unwrap()["samples"][0]["cpu_reference_ns"].is_null());
    let mut measured = Case::new("matmul", "rust", vec![2, 2, 2]);
    measured.samples.push(sample(Some(7)));
    assert_eq!(
        measured.finish().unwrap().summary_ns["cpu_reference_ns"].median,
        7
    );
}
