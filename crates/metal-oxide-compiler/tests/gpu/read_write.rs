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
            let submission = unsafe {
                device.submit(|batch| {
                    for _ in 0..2 {
                        batch.launch(
                            &pipeline,
                            LaunchConfig::<32>::for_elements(pairs)?,
                            &[Argument::read_write(&mut values), Argument::value(pairs)?],
                        )?;
                    }
                    Ok(())
                })?
            };
            match completion {
                0 => {
                    submission.wait()?;
                }
                1 => drop(submission),
                _ => std::mem::forget(submission),
            }
            for (index, (&actual, &initial)) in values.as_slice().iter().zip(&input).enumerate() {
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
            values.as_mut_slice().copy_from_slice(&input);
            assert_eq!(values.as_slice(), input);
        }
    }
    Ok(())
}
