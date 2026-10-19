use metal_oxide::{Device, Error, LaunchConfig, Operation};

#[test]
#[ignore = "requires a Metal device"]
fn saxpy_matches_cpu() {
    super::main().unwrap();
}

#[test]
#[ignore = "requires a Metal device"]
fn unequal_lengths_violate_the_kernel_requirement() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let x = device.buffer_from_slice(&[1.0_f32, 2.0])?;
    let mut y = device.buffer_from_slice(&[1.0_f32; 3])?;
    let config = LaunchConfig::<256>::for_elements(3)?;
    let result = super::saxpy(&device, config, 2.0, &x, &mut y).wait();
    assert!(matches!(
        result,
        Err(Error::Contract { relation, .. }) if relation == "x.len() == y.len()"
    ));
    assert_eq!(y.read().wait(), [1.0; 3]);
    Ok(())
}
