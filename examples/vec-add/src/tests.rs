use metal_oxide::{Device, Error, LaunchConfig, Operation};

#[test]
#[ignore = "requires a Metal device"]
fn vec_add_matches_cpu() {
    super::main().unwrap();
}

#[test]
#[ignore = "requires a Metal device"]
fn unequal_lengths_violate_the_kernel_requirements() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let a = device.buffer_from_slice(&[1.0_f32, 2.0])?;
    let mut out = device.buffer_zeroed::<f32>(3)?;
    let config = LaunchConfig::<256>::for_elements(3)?;
    let result = super::vec_add(&device, config, &a, &a, &mut out).wait();
    assert!(matches!(
        result,
        Err(Error::Contract { relation, .. }) if relation == "a.len() == out.len()"
    ));
    assert_eq!(out.read().wait(), [0.0; 3]);
    Ok(())
}
