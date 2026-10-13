use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn dynamic_shared_records_reduce_variable_blocks_and_partial_inputs() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/dynamic_shared.rs",
        "dynamic_reduce",
    )?;
    for block in [1, 7, 32, 64, 127] {
        for n in [0, 1, block - 1, block, block + 1, block * 3 - 1] {
            let input = (0..n).map(|i| i as f32 * 0.25).collect::<Vec<_>>();
            let values = device.buffer_from_slice(&input)?;
            let config = DynamicLaunchConfig::for_elements(n, block)?;
            let mut output =
                device.buffer_from_slice(&vec![-8192.0_f32; config.grid.x as usize + 3])?;
            // SAFETY: each block has block elements in both scratch allocations. All
            // lanes initialize their elements and participate in the helper barrier.
            unsafe {
                Launch::new(
                    Ok(&pipeline),
                    config,
                    (&values, &mut output, block as usize, n),
                    |(values, output, block, n)| {
                        Ok([
                            Argument::read(values),
                            Argument::threadgroup::<f32>(*block)?,
                            Argument::value(0.5_f32)?,
                            Argument::write(output),
                            Argument::threadgroup::<(u8, u64)>(*block)?,
                            Argument::value(*n)?,
                        ])
                    },
                )
            }
            .wait()?;
            for (group, actual) in output.read().wait()[..config.grid.x as usize]
                .iter()
                .enumerate()
            {
                let expected = (group as u32 * block..((group as u32 + 1) * block).min(n))
                    .map(|i| input[i as usize] + 0.5 + (i % 17) as f32 + i as f32)
                    .sum::<f32>()
                    + 7.0;
                assert_eq!(*actual, expected, "n={n}, block={block}, group={group}");
            }
            assert!(
                output.read().wait()[config.grid.x as usize..]
                    .iter()
                    .all(|&x| x == -8192.0)
            );
        }
    }
    Ok(())
}
