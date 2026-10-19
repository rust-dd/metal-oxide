use super::*;

#[test]
#[ignore = "requires a Metal device"]
fn generated_gemm_matches_cpu() {
    use objc2_metal::{MTLDevice, MTLGPUFamily};
    let device = metal_oxide::Device::system_default().unwrap();
    // Apple7 and later GPUs must run the kernels; only other devices may reject them.
    let apple7 = device.as_native().supportsFamily(MTLGPUFamily::Apple7);
    assert_eq!(verify(&device).unwrap(), apple7);
}
