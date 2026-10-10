#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use metal_oxide::{Argument, Device, Error, LaunchConfig, Module, Pipeline};
use objc2_metal::{MTLCopyAllDevices, MTLDevice};

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn enumerated_devices_have_native_budgets_and_independent_contexts() -> metal_oxide::Result<()> {
    let native = MTLCopyAllDevices();
    let devices = Device::all()?;
    assert_eq!(devices.len(), native.len());
    assert!(!devices.is_empty());
    for (device, raw) in devices.iter().zip(native.iter()) {
        assert_eq!(device.name(), raw.name().to_string());
        assert_eq!(
            device.recommended_max_working_set_size(),
            raw.recommendedMaxWorkingSetSize()
        );
        assert!(device.recommended_max_working_set_size() > 0);
    }
    let count = devices.len();
    assert!(Device::all()?.into_iter().nth(count).is_none());
    let selected = devices.into_iter().next().unwrap();
    let other = Device::all()?.into_iter().next().unwrap();
    let source = "#include <metal_stdlib>\nusing namespace metal;\nkernel void fill(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) { out[i] = i; }";
    let module = Module::from_source(&selected, source)?;
    assert!(matches!(
        Pipeline::new(&other, &module, "fill"),
        Err(Error::DeviceMismatch)
    ));
    let pipeline = Pipeline::new(&selected, &module, "fill")?;
    let mut output = selected.buffer_zeroed::<u32>(32)?;
    // SAFETY: one block writes 32 distinct initialized elements.
    unsafe {
        selected.launch(
            &pipeline,
            LaunchConfig::<32>::for_elements(32)?,
            &[Argument::write(&mut output)],
        )?
    };
    assert_eq!(output.as_slice(), (0..32).collect::<Vec<_>>());
    Ok(())
}
