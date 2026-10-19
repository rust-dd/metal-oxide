use super::*;

#[test]
fn a_rejected_library_or_pipeline_becomes_the_reason_to_skip() {
    let rejected = |error| accepted::<()>(Err(error)).unwrap().unwrap_err();
    let library = rejected(metal_oxide::Error::Library("no tensors".into()));
    assert_eq!(library, "Metal library error: no tensors");
    let pipeline = rejected(metal_oxide::Error::Pipeline("unsupported".into()));
    assert_eq!(pipeline, "Metal pipeline error: unsupported");
}

#[test]
fn a_loaded_pipeline_passes_and_other_errors_stay_errors() {
    assert_eq!(accepted(Ok(7)).unwrap(), Ok(7));
    let missing = accepted::<()>(Err(metal_oxide::Error::KernelNotFound("gemm".into())));
    assert_eq!(
        missing.unwrap_err().to_string(),
        "Metal kernel \"gemm\" was not found"
    );
    assert!(accepted::<()>(Err(metal_oxide::Error::InvalidPath)).is_err());
}

#[test]
fn the_grid_tiles_c_as_the_kernel_source_does() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = std::fs::read_to_string(root.join("../reference-msl/gemm-mpp.metal")).unwrap();
    let constant = |name: &str| {
        let line = format!("constant constexpr int {name} = ");
        let (_, rest) = source.split_once(&line).unwrap();
        rest.split_once(';').unwrap().0.parse::<u32>().unwrap()
    };
    assert_eq!(TILE, [constant("ROWS"), constant("COLUMNS")]);
}
