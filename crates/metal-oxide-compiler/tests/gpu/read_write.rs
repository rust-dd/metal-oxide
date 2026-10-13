use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn in_place_fft_buffers_survive_ordered_and_cancelled_submissions() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/read_write.rs",
        "fft_stage",
    )?;
    for pairs in [0_u32, 1, 31, 32, 33, 100_003] {
        let input = (0..pairs * 2)
            .map(|i| (i % 8192) as f32 - 4096.0)
            .chain([12345.0; 64])
            .collect::<Vec<_>>();
        for completion in 0..3 {
            let mut values = device.buffer_from_slice(&input)?;
            // SAFETY: each lane owns a distinct pair; the ordered second stage reads
            // the first stage's results. Both dispatches stay within the allocation.
            let operation = sequence(|batch| unsafe {
                for _ in 0..2 {
                    batch.push(Launch::new(
                        Ok(&pipeline),
                        LaunchConfig::<32>::for_elements(pairs)?,
                        (&mut values, pairs),
                        |(values, pairs)| {
                            Ok([Argument::read_write(values), Argument::value(*pairs)?])
                        },
                    ))?;
                }
                Ok(())
            });
            match completion {
                0 => {
                    operation.wait()?;
                }
                1 => drop(started(operation)),
                _ => std::mem::forget(started(operation)),
            }
            for (index, (&actual, &initial)) in values.read().wait().iter().zip(&input).enumerate()
            {
                let expected = if index < pairs as usize * 2 {
                    initial * 2.0
                } else {
                    initial
                };
                assert_eq!(
                    actual, expected,
                    "pairs={pairs}, completion={completion}, index={index}"
                );
            }
            values.write().wait().copy_from_slice(&input);
            assert_eq!(values.read().wait(), input);
        }
    }
    Ok(())
}
