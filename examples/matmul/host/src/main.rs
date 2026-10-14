#[metal_oxide::kernels(path = "../kernels")]
mod kernels {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod mixed;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = metal_oxide::Device::system_default()?;
    verify_f32(&device)?;
    mixed::verify(&device)?;
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn verify_f32(device: &metal_oxide::Device) -> metal_oxide::Result<()> {
    use metal_oxide::{Dim3, LaunchConfig, Operation};
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
        unsafe { kernels::dot(device, config, &input_a, &input_b, &mut output, n) }.wait()?;
        assert_eq!(
            output.read().wait().iter().sum::<f32>(),
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
            kernels::matmul(
                device,
                LaunchConfig::<16, 16>::new(Dim3::xy(columns.div_ceil(16), rows.div_ceil(16))),
                &input_a,
                &input_b,
                &mut output,
                rows,
                columns,
                inner,
            )
        }
        .wait()?;
        for row in 0..rows {
            for column in 0..columns {
                let expected = (0..inner)
                    .map(|k| a[(row * inner + k) as usize] * b[(k * columns + column) as usize])
                    .sum::<f32>();
                assert_eq!(
                    output.read().wait()[(row * columns + column) as usize],
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

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("matmul runs on macOS with Apple Silicon");
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a Metal device"]
fn generated_matmul_matches_cpu() {
    let device = metal_oxide::Device::system_default().unwrap();
    verify_f32(&device).unwrap();
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a Metal device"]
fn generated_half_matmul_matches_cpu_and_msl() {
    let device = metal_oxide::Device::system_default().unwrap();
    mixed::verify(&device).unwrap();
}
