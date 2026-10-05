use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn reduction_matches_cpu_at_block_boundaries() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/cooperative.rs",
        "reduce",
    )?;
    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        let values = (0..n).map(|i| (i % 17) as f32 - 8.0).collect::<Vec<_>>();
        let input = device.buffer_from_slice(&values)?;
        let config = LaunchConfig::<256>::for_elements(n)?;
        let mut output = device.buffer_zeroed::<f32>(n.div_ceil(256) as usize)?;
        // SAFETY: input covers n; each 256-thread block owns one initialized shared tile and output.
        unsafe {
            device.launch(
                &pipeline,
                config,
                &[
                    Argument::read(&input),
                    Argument::write(&mut output),
                    Argument::u32(n),
                ],
            )?;
        }
        let expected = values
            .chunks(256)
            .map(|v| v.iter().sum::<f32>())
            .collect::<Vec<_>>();
        assert_eq!(output.as_slice(), expected, "n={n}");
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn tiled_transpose_matches_cpu_on_rectangles_and_partial_tiles() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/cooperative.rs",
        "transpose",
    )?;
    for (width, height) in [(1_u32, 1_u32), (16, 16), (17, 31), (255, 257)] {
        let values = (0..width * height).map(|i| i as f32).collect::<Vec<_>>();
        let input = device.buffer_from_slice(&values)?;
        let mut output = device.buffer_zeroed::<f32>(values.len())?;
        // SAFETY: distinct full-sized buffers; 16x16 blocks initialize all shared cells before use.
        unsafe {
            device.launch(
                &pipeline,
                LaunchConfig::<16, 16>::new(Dim3::xy(width.div_ceil(16), height.div_ceil(16))),
                &[
                    Argument::read(&input),
                    Argument::write(&mut output),
                    Argument::u32(width),
                    Argument::u32(height),
                ],
            )?;
        }
        for y in 0..height {
            for x in 0..width {
                assert_eq!(
                    output.as_slice()[(x * height + y) as usize],
                    values[(y * width + x) as usize]
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn contended_atomic_add_counts_every_thread() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/cooperative.rs",
        "counter",
    )?;
    let mut output = device.buffer_zeroed::<u32>(1)?;
    let config = LaunchConfig::<256>::new(Dim3::x(257));
    // SAFETY: all 65792 threads atomically update the same initialized u32.
    unsafe {
        device.launch(&pipeline, config, &[Argument::atomic(&mut output)])?;
    }
    assert_eq!(output.as_slice(), [256 * 257]);
    Ok(())
}
