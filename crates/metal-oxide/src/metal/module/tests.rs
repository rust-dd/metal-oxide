use super::*;

fn apple(newest: MTLGPUFamily) -> impl Fn(MTLGPUFamily) -> bool {
    move |family| (MTLGPUFamily::Apple1.0..=newest.0).contains(&family.0)
}

#[test]
fn bfloat_requires_an_apple_family_that_supports_it() {
    assert!(require("bfloat16", apple(MTLGPUFamily::Apple6)).is_ok());
    assert!(matches!(
        require("bfloat16", apple(MTLGPUFamily::Apple5)),
        Err(Error::UnsupportedDevice(_))
    ));
    assert!(matches!(
        require("bfloat16", |family| family == MTLGPUFamily::Mac2),
        Err(Error::UnsupportedDevice(_))
    ));
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
