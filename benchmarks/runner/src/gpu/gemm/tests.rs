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
    assert_eq!(case.block, Some([128, 1, 1]));
    assert!(!case.summary_ns.contains_key("cpu_reference_ns"));
    assert!(serde_json::to_value(&case).unwrap()["samples"][0]["cpu_reference_ns"].is_null());
    let mut measured = Case::new("matmul", "rust", vec![2, 2, 2]);
    measured.samples.push(sample(Some(7)));
    assert_eq!(
        measured.finish().unwrap().summary_ns["cpu_reference_ns"].median,
        7
    );
}

#[test]
fn problem_holds_its_inputs_exactly_in_f16() {
    let problem = Problem::new([17, 13, 3]);
    let [a, b] = problem.half();
    let widen = |values: &[F16]| values.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    assert_eq!(widen(&a), problem.a);
    assert_eq!(widen(&b), problem.b);
}

#[test]
fn finish_adds_the_gpu_time_of_one_gemm_rounded_to_the_nanosecond() {
    let mut case = Case::new("gemm-f32", "mps", vec![2, 2, 2]);
    case.repeat = 4;
    for gpu_ns in [400, 1000, 2002] {
        case.samples.push(Sample {
            gpu_ns: Some(gpu_ns),
            ..Sample::default()
        });
    }
    let case = finish(case).unwrap();
    assert_eq!(case.summary_ns["gpu_ns"].median, 1000);
    let each = &case.summary_ns["gpu_per_gemm_ns"];
    assert_eq!((each.min, each.median, each.max), (100, 250, 501));
}

#[test]
fn finish_reports_no_gpu_time_per_gemm_without_gpu_timestamps() {
    let mut case = Case::new("gemm-f32", "rust", vec![2, 2, 2]);
    case.samples.push(Sample::default());
    let case = finish(case).unwrap();
    assert!(!case.summary_ns.contains_key("gpu_per_gemm_ns"));
}

#[test]
fn finish_adds_the_host_time_of_one_gemm_from_submit_and_wait() {
    let mut case = Case::new("gemm-f16", "msl", vec![2, 2, 2]);
    case.repeat = 4;
    for (submit_ns, wait_ns) in [(100, 300), (250, 750), (2, 2000)] {
        case.samples.push(Sample {
            submit_ns,
            wait_ns,
            encode_ns: 50,
            readback_ns: 900,
            ..Sample::default()
        });
    }
    let case = finish(case).unwrap();
    let each = &case.summary_ns["host_per_gemm_ns"];
    assert_eq!((each.min, each.median, each.max), (100, 250, 501));
    assert!(!case.summary_ns.contains_key("gpu_per_gemm_ns"));
}
