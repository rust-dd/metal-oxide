use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn block_scan_handles_padded_threads_and_wrapping_values() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "examples/cooperation/kernels/src/lib.rs",
        "block_scan",
    )?;
    for n in [1_u32, 255, 256, 257, 1025] {
        let values = (0..n)
            .map(|i| if i % 3 == 0 { u32::MAX } else { i })
            .collect::<Vec<_>>();
        let input = device.buffer_from_slice(&values)?;
        let mut prefix = device.buffer_zeroed::<u32>(n as usize)?;
        let mut totals = device.buffer_zeroed::<u32>(n.div_ceil(256) as usize)?;
        // SAFETY: complete 256-thread blocks participate; guarded reads use neutral zero padding.
        unsafe {
            device.launch(
                &pipeline,
                LaunchConfig::<256>::for_elements(n)?,
                &[
                    Argument::read(&input),
                    Argument::write(&mut prefix),
                    Argument::write(&mut totals),
                    Argument::value(n)?,
                ],
            )?;
        }
        for (block, chunk) in values.chunks(256).enumerate() {
            let mut sum = 0_u32;
            for (lane, &value) in chunk.iter().enumerate() {
                assert_eq!(
                    prefix.as_slice()[block * 256 + lane],
                    sum,
                    "n={n}, block={block}, lane={lane}"
                );
                sum = sum.wrapping_add(value);
            }
            assert_eq!(totals.as_slice()[block], sum);
        }
    }
    Ok(())
}
