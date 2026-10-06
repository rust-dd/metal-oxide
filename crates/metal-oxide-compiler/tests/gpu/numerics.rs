use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn simd_shuffle_broadcasts_each_group_source_lane() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/simd_shuffle.rs",
        "broadcast",
    )?;
    let width = pipeline.thread_execution_width();
    let mut output = device.buffer_zeroed::<f32>(3 * 256)?;
    // SAFETY: full blocks participate, lane zero is active, and output covers every dispatched thread.
    unsafe {
        device.launch(
            &pipeline,
            LaunchConfig::<256>::new(Dim3::x(3)),
            &[Argument::write(&mut output)],
        )?;
    }
    for (i, &value) in output.as_slice().iter().enumerate() {
        assert_eq!(value, (i / width * width) as f32);
    }
    Ok(())
}

fn reference(device: &Device, name: &str) -> metal_oxide::Result<Pipeline> {
    let module = Module::from_source(
        device,
        include_str!("../../../../benchmarks/reference-msl/numerics.metal"),
    )?;
    Pipeline::new(device, &module, name)
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn simd_dot_matches_cpu_and_reference_msl() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let generated = pipeline(&device, "examples/matmul/kernels/src/lib.rs", "dot")?;
    let reference = reference(&device, "dot")?;
    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        let a = (0..n)
            .map(|i| (i % 13) as f32 * 0.25 - 1.0)
            .collect::<Vec<_>>();
        let b = (0..n)
            .map(|i| (i % 7) as f32 * 0.5 - 2.0)
            .collect::<Vec<_>>();
        let input_a = device.buffer_from_slice(&a)?;
        let input_b = device.buffer_from_slice(&b)?;
        let config = LaunchConfig::<256>::for_elements(n)?;
        let groups = 256_usize.div_ceil(generated.thread_execution_width());
        let mut results = Vec::new();
        for pipeline in [&generated, &reference] {
            let mut output = device.buffer_zeroed::<f32>(config.grid.x as usize * groups)?;
            // SAFETY: buffers cover n; full 256-thread blocks participate and one lane writes each partial.
            unsafe {
                device.launch(
                    pipeline,
                    config,
                    &[
                        Argument::read(&input_a),
                        Argument::read(&input_b),
                        Argument::write(&mut output),
                        Argument::u32(n),
                    ],
                )?;
            }
            results.push(output.as_slice().to_vec());
        }
        assert_eq!(results[0], results[1]);
        assert_eq!(
            results[0].iter().sum::<f32>(),
            a.iter().zip(&b).map(|(a, b)| a * b).sum::<f32>()
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn tiled_matmul_matches_cpu_and_reference_msl() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let generated = pipeline(&device, "examples/matmul/kernels/src/lib.rs", "matmul")?;
    let reference = reference(&device, "matmul")?;
    for (rows, columns, inner) in [
        (0_u32, 5_u32, 7_u32),
        (1, 1, 1),
        (16, 16, 16),
        (17, 31, 19),
        (64, 48, 33),
        (5, 7, 0),
    ] {
        let a = (0..rows * inner)
            .map(|i| (i % 13) as f32 * 0.25 - 1.0)
            .collect::<Vec<_>>();
        let b = (0..inner * columns)
            .map(|i| (i % 7) as f32 * 0.5 - 2.0)
            .collect::<Vec<_>>();
        let input_a = device.buffer_from_slice(&a)?;
        let input_b = device.buffer_from_slice(&b)?;
        let config = LaunchConfig::<16, 16>::new(Dim3::xy(columns.div_ceil(16), rows.div_ceil(16)));
        let expected = (0..rows)
            .flat_map(|r| (0..columns).map(move |c| (r, c)))
            .map(|(r, c)| {
                (0..inner)
                    .map(|k| a[(r * inner + k) as usize] * b[(k * columns + c) as usize])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        for pipeline in [&generated, &reference] {
            let mut output = device.buffer_zeroed::<f32>((rows * columns) as usize)?;
            // SAFETY: buffers have the declared matrix dimensions, with uniform tile barriers and one output owner.
            unsafe {
                device.launch(
                    pipeline,
                    config,
                    &[
                        Argument::read(&input_a),
                        Argument::read(&input_b),
                        Argument::write(&mut output),
                        Argument::u32(rows),
                        Argument::u32(columns),
                        Argument::u32(inner),
                    ],
                )?;
            }
            assert_eq!(output.as_slice(), expected);
        }
    }
    Ok(())
}
