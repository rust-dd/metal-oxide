#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use metal_oxide::{Argument, Device, Dispatch1d, Module, Pipeline};

const SOURCE: &str = include_str!("../../../examples/vec-add/kernels/vec_add.metal");

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn vec_add_matches_cpu_at_dispatch_boundaries() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "vec_add")?;

    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        let a = (0..n).map(|i| (i % 1024) as f32 * 0.25).collect::<Vec<_>>();
        let b = (0..n).map(|i| (i % 511) as f32 * -0.5).collect::<Vec<_>>();
        let input_a = device.buffer_from_slice(&a)?;
        let input_b = device.buffer_from_slice(&b)?;
        let mut output = device.buffer_zeroed::<f32>(n as usize)?;

        // SAFETY: buffers have n elements, distinct allocations, and one writer per index.
        unsafe {
            device.dispatch(
                &pipeline,
                Dispatch1d::new(n).with_group_width(256),
                &[
                    Argument::read(&input_a),
                    Argument::read(&input_b),
                    Argument::write(&mut output),
                    Argument::u32(n),
                ],
            )?;
        }

        for (i, actual) in output.as_slice().iter().enumerate() {
            assert_eq!(*actual, a[i] + b[i], "n={n}, index={i}");
        }
    }

    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn typed_buffers_are_initialized_and_mutable_on_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let mut buffer = device.buffer_zeroed::<u32>(3)?;
    assert_eq!(buffer.as_slice(), [0, 0, 0]);
    buffer.as_mut_slice().copy_from_slice(&[1, 2, 3]);
    assert_eq!(buffer.as_slice(), [1, 2, 3]);
    assert!(device.buffer_zeroed::<i32>(0)?.is_empty());
    assert!(device.buffer_zeroed::<u32>(usize::MAX).is_err());
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn invalid_shader_and_missing_entrypoint_return_errors() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    assert!(Module::from_source(&device, "this is not Metal source").is_err());
    let module = Module::from_source(&device, SOURCE)?;
    assert!(Pipeline::new(&device, &module, "missing").is_err());
    Ok(())
}
