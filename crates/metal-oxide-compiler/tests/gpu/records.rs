use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn owned_struct_helpers_and_field_updates_match_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/records.rs",
        "record_math",
    )?;
    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        let a = (0..n)
            .map(|i| (i % 13) as f32 * 0.25 - 1.0)
            .collect::<Vec<_>>();
        let b = (0..n)
            .map(|i| (i % 7) as f32 * 0.5 - 2.0)
            .collect::<Vec<_>>();
        let input_a = device.buffer_from_slice(&a)?;
        let input_b = device.buffer_from_slice(&b)?;
        let mut sum = device.buffer_zeroed::<f32>(n as usize)?;
        let mut product = device.buffer_zeroed::<f32>(n as usize)?;
        // SAFETY: distinct buffers cover n, and each active thread owns one element in each output.
        unsafe {
            device.launch(
                &pipeline,
                LaunchConfig::<256>::for_elements(n)?,
                &[
                    Argument::read(&input_a),
                    Argument::read(&input_b),
                    Argument::write(&mut sum),
                    Argument::write(&mut product),
                    Argument::value::<u32>(n)?,
                ],
            )?;
        }
        for i in 0..n as usize {
            let expected = (a[i] + b[i]) * 3.0;
            assert_eq!(sum.as_slice()[i], expected);
            assert_eq!(product.as_slice()[i], a[i] * b[i] + expected);
        }
    }
    Ok(())
}
