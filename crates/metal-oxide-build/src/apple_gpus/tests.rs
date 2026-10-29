use super::supports;

#[test]
fn apple_gpus_run_the_families_their_generation_supports() {
    assert!(supports("applegpu_g13g", 7));
    assert!(!supports("applegpu_g13d", 8));
    assert!(supports("applegpu_g14s", 8));
    assert!(supports("applegpu_g16s", 9));
    assert!(supports("applegpu_g17p", 9));
    assert!(!supports("applegpu_g16g", 10));
}

#[test]
fn other_offline_targets_are_not_apple_gpus() {
    assert!(!supports("amdgpu_gfx1030", 1));
    assert!(!supports("intelgpu_icl_1x8x8r7", 1));
    assert!(!supports("applegpu_gXg", 1));
}
