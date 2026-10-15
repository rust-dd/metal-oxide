use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn requested_backend_uses_its_native_command_queue() {
    let device = crate::Device::system_default().unwrap();
    match std::env::var("METAL_OXIDE_BACKEND").as_deref() {
        Ok("metal4") => assert!(matches!(device.backend(), Backend::Metal4 { .. })),
        Ok("classic") => assert!(matches!(device.backend(), Backend::Classic { .. })),
        _ => assert_eq!(
            matches!(device.backend(), Backend::Metal4 { .. }),
            Backend::supports_metal4(device.raw())
        ),
    }
}
