use super::*;

#[test]
#[ignore = "requires a Metal device"]
fn generated_matmul_matches_cpu() {
    let device = metal_oxide::Device::system_default().unwrap();
    verify_f32(&device).unwrap();
}

#[test]
#[ignore = "requires a Metal device"]
fn generated_half_matmul_matches_cpu_and_msl() {
    let device = metal_oxide::Device::system_default().unwrap();
    mixed::verify(&device).unwrap();
}
