use super::*;

fn apple(newest: MTLGPUFamily) -> impl Fn(MTLGPUFamily) -> bool {
    move |family| (MTLGPUFamily::Apple1.0..=newest.0).contains(&family.0)
}

#[test]
fn apple_or_mac2_families_satisfy_artifact_features() {
    for feature in [
        "simd_groups",
        "int32_atomics",
        "dynamic_threadgroup_memory",
        "logging",
    ] {
        assert!(require(feature, apple(MTLGPUFamily::Apple7)).is_ok());
        assert!(require(feature, |family| family == MTLGPUFamily::Mac2).is_ok());
        assert!(matches!(
            require(feature, |_| false),
            Err(Error::UnsupportedDevice(_))
        ));
    }
    assert!(matches!(
        require("simd_groups", apple(MTLGPUFamily::Apple6)),
        Err(Error::UnsupportedDevice(_))
    ));
}
