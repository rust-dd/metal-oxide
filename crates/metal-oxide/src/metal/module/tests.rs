use super::*;

fn apple(newest: MTLGPUFamily) -> impl Fn(MTLGPUFamily) -> bool {
    move |family| (MTLGPUFamily::Apple1.0..=newest.0).contains(&family.0)
}

fn mac2(family: MTLGPUFamily) -> bool {
    family == MTLGPUFamily::Mac2
}

#[test]
fn bfloat_requires_an_apple_family_that_supports_it() {
    assert!(require("bfloat16", 15, apple(MTLGPUFamily::Apple6)).is_ok());
    assert!(matches!(
        require("bfloat16", 15, apple(MTLGPUFamily::Apple5)),
        Err(Error::UnsupportedDevice(_))
    ));
    assert!(matches!(
        require("bfloat16", 15, mac2),
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
        assert!(require(feature, 15, apple(MTLGPUFamily::Apple7)).is_ok());
        assert!(require(feature, 15, mac2).is_ok());
        assert!(matches!(
            require(feature, 15, |_| false),
            Err(Error::UnsupportedDevice(_))
        ));
    }
    assert!(matches!(
        require("simd_groups", 15, apple(MTLGPUFamily::Apple6)),
        Err(Error::UnsupportedDevice(_))
    ));
}

#[test]
fn atomic_features_check_os_and_apple_family() {
    for (name, macos, family) in [
        ("float32_atomics", 13, MTLGPUFamily::Apple7),
        ("uint64_atomics", 12, MTLGPUFamily::Apple8),
        ("atomic_acquire_release", 27, MTLGPUFamily::Apple7),
    ] {
        assert!(require(name, macos, apple(family)).is_ok());
        assert!(require(name, macos - 1, apple(family)).is_err());
        assert!(require(name, macos, apple(MTLGPUFamily(family.0 - 1))).is_err());
        assert!(require(name, macos, mac2).is_err());
    }
}

#[test]
fn simdgroup_matrices_require_apple7_and_32_lane_simd_groups() {
    assert!(require("simdgroup_matrix", 15, apple(MTLGPUFamily::Apple7)).is_ok());
    assert!(matches!(
        require("simdgroup_matrix", 15, apple(MTLGPUFamily::Apple6)),
        Err(Error::UnsupportedDevice(_))
    ));
    assert!(matches!(
        require("simdgroup_matrix", 15, mac2),
        Err(Error::UnsupportedDevice(_))
    ));
    let matrices = ["simd_groups".to_owned(), "simdgroup_matrix".to_owned()];
    assert!(require_simd_width(&matrices, 32).is_ok());
    assert!(matches!(
        require_simd_width(&matrices, 64),
        Err(Error::UnsupportedDevice(reason)) if reason.contains("32-lane")
    ));
    assert!(require_simd_width(&matrices[..1], 64).is_ok());
}
