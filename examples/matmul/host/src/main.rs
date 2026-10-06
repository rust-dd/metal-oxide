#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod kernels {
    include!(env!("METAL_OXIDE_BINDINGS"));
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use metal_oxide::{Device, Dim3, LaunchConfig};
    let device = Device::system_default()?;
    let kernels = kernels::load(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
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
        let mut output = device.buffer_zeroed::<f32>(config.grid.x as usize * 256)?;
        // SAFETY: distinct inputs cover n; output has space even for one SIMD group per thread.
        unsafe {
            kernels.dot(config, &input_a, &input_b, &mut output, n)?;
        }
        assert_eq!(
            output.as_slice().iter().sum::<f32>(),
            a.iter().zip(&b).map(|(a, b)| a * b).sum::<f32>()
        );
        println!("dot n={n}: verified on {}", device.name());
    }
    for (rows, columns, inner) in [
        (1_u32, 1_u32, 1_u32),
        (16, 16, 16),
        (17, 31, 19),
        (64, 48, 33),
    ] {
        let a = (0..rows * inner)
            .map(|i| (i % 13) as f32 * 0.25 - 1.0)
            .collect::<Vec<_>>();
        let b = (0..inner * columns)
            .map(|i| (i % 7) as f32 * 0.5 - 2.0)
            .collect::<Vec<_>>();
        let input_a = device.buffer_from_slice(&a)?;
        let input_b = device.buffer_from_slice(&b)?;
        let mut output = device.buffer_zeroed::<f32>((rows * columns) as usize)?;
        // SAFETY: the grid covers all output cells; distinct buffers have the documented matrix sizes.
        unsafe {
            kernels.matmul(
                LaunchConfig::<16, 16>::new(Dim3::xy(columns.div_ceil(16), rows.div_ceil(16))),
                &input_a,
                &input_b,
                &mut output,
                rows,
                columns,
                inner,
            )?;
        }
        for row in 0..rows {
            for column in 0..columns {
                let expected = (0..inner)
                    .map(|k| a[(row * inner + k) as usize] * b[(k * columns + column) as usize])
                    .sum::<f32>();
                assert_eq!(
                    output.as_slice()[(row * columns + column) as usize],
                    expected
                );
            }
        }
        println!(
            "matmul {rows}x{inner} x {inner}x{columns}: verified on {}",
            device.name()
        );
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact)))]
fn main() -> std::process::ExitCode {
    eprintln!("run with cargo metal run -p matmul on macOS Apple Silicon");
    std::process::ExitCode::FAILURE
}

#[cfg(all(
    test,
    target_os = "macos",
    target_arch = "aarch64",
    metal_oxide_artifact
))]
#[test]
#[ignore = "requires a Metal device and generated artifact"]
fn generated_matmul_matches_cpu() {
    main().unwrap();
}
