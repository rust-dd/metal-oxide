use super::{INPUTS, Saved};
use crate::recorded::Recorded;
use metal_oxide_artifact::{ABI_VERSION, MathMode};
use std::path::Path;

/// Published crates ship saved artifacts; an incompatible change must increment
/// `ABI_VERSION` and replace this fixture with the new release's artifact.
#[test]
fn alpha0_saved_artifacts_load_like_a_published_dependency() {
    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/alpha0");
    let saved = Saved::new(
        &package,
        Path::new("src/main.rs"),
        "vec_add",
        MathMode::Precise,
    );
    let inputs = std::fs::read(saved.directory.join(INPUTS)).unwrap();
    let inputs = serde_json::from_slice::<Recorded>(&inputs).unwrap();
    let built = saved.load(&inputs.request, None).unwrap();
    assert_eq!(built.abi.version, ABI_VERSION);
    assert_eq!(built.abi.kernels[0].name, "vec_add");
    assert!(built.saved);
}
