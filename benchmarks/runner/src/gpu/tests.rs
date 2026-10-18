use super::*;

fn config(gemm_repeat: u32) -> Config {
    serde_json::from_value(serde_json::json!({
        "artifacts": {}, "references": {}, "samples": 1, "warmup": 0, "elements": 1,
        "matrix": [1, 1, 1], "gemm": [1, 1, 1], "gemm_repeat": gemm_repeat,
    }))
    .unwrap()
}

#[test]
fn report_requires_samples_and_keeps_unavailable_gpu_time_absent() {
    assert!(Case::new("vec-add", "rust", vec![1]).finish().is_err());
    let mut case = Case::new("vec-add", "rust", vec![1]);
    case.samples.push(Sample {
        submit_ns: 100,
        ..Sample::default()
    });
    let report = case.finish().unwrap();
    assert!(!report.summary_ns.contains_key("gpu_ns"));
    assert_eq!(report.summary_ns["submit_ns"].median, 100);
    assert!(serde_json::to_value(&report).unwrap()["samples"][0]["gpu_ns"].is_null());
}

#[test]
fn cases_run_once_per_sample_by_default() {
    let report = serde_json::to_value(Case::new("scan", "msl", vec![1])).unwrap();
    assert_eq!(report["repeat"], 1);
    assert_eq!(report["block"], serde_json::json!([256, 1, 1]));
}

#[test]
fn config_accepts_a_gemm_repeat_from_1_to_256_only() {
    for repeat in [1, 8, 256] {
        config(repeat).validate().unwrap();
    }
    for repeat in [0, 257] {
        let error = config(repeat).validate().unwrap_err().to_string();
        assert_eq!(error, "gemm repeat must be from 1 to 256");
    }
}

#[test]
fn config_keeps_the_established_bounds() {
    let mut zero_samples = config(1);
    zero_samples.samples = 0;
    assert!(zero_samples.validate().is_err());
    let mut tall = config(1);
    tall.gemm = [1, 8193, 1];
    assert!(tall.validate().is_err());
    let mut wide = config(1);
    wide.matrix = [1, 1025, 1];
    assert!(wide.validate().is_err());
}

#[test]
fn mps_is_a_backend_of_its_own_in_the_classic_run_only() {
    let mps = || Ok(vec![Case::new("gemm-f32", "mps", vec![1, 1, 1])]);
    let classic = backend_reports("GPU", "classic", vec![], mps).unwrap();
    let names = |reports: &[serde_json::Value]| {
        reports
            .iter()
            .map(|report| report["backend"].as_str().unwrap().to_string())
            .collect::<Vec<_>>()
    };
    assert_eq!(names(&classic), ["classic", "mps"]);
    assert_eq!(classic[1]["cases"][0]["variant"], "mps");
    assert_eq!(classic[1]["device"], "GPU");
    let metal4 = backend_reports("GPU", "metal4", vec![], || panic!("MPS ran under Metal 4"));
    assert_eq!(names(&metal4.unwrap()), ["metal4"]);
}
