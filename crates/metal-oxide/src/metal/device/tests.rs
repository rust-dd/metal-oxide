use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn the_default_context_lives_as_long_as_one_of_its_clones() -> Result<()> {
    let first = Device::system_default()?;
    let second = Device::system_default()?;
    assert_eq!(first.id(), second.id());
    let independent = Device::all()?.remove(0);
    assert_ne!(independent.id(), first.id());
    let id = first.id();
    drop((first, second));
    assert_ne!(Device::system_default()?.id(), id);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn the_native_api_follows_the_backend_selection() -> Result<()> {
    let device = Device::system_default()?;
    let metal4 = match std::env::var("METAL_OXIDE_BACKEND").as_deref() {
        Ok("classic") => false,
        Ok("metal4") => true,
        _ => Backend::supports_metal4(device.raw()),
    };
    let (api, name) = if metal4 {
        (NativeApi::Metal4, "Metal 4")
    } else {
        (NativeApi::Classic, "classic Metal")
    };
    assert_eq!(device.native_api(), api);
    assert_eq!(device.native_api().to_string(), name);
    Ok(())
}
