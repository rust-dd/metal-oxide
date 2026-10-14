#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod kernels {
    include!(env!("METAL_OXIDE_BINDINGS"));
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use metal_oxide::{Device, LaunchConfig, Operation};

    let device = Device::system_default()?;
    let kernels = kernels::load(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
    println!("Device: {}", device.name());

    let a = 2.5_f32;
    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        let x = (0..n).map(|i| (i % 1024) as f32 * 0.25).collect::<Vec<_>>();
        let y = (0..n).map(|i| (i % 511) as f32 * -0.5).collect::<Vec<_>>();
        let input = device.buffer_from_slice(&x)?;
        let mut output = device.buffer_from_slice(&y)?;
        kernels
            .saxpy(
                LaunchConfig::<256>::for_elements(n)?,
                a,
                &input,
                &mut output,
            )
            .wait()?;
        for (i, actual) in output.read().wait().iter().enumerate() {
            if *actual != a * x[i] + y[i] {
                return Err(format!("saxpy mismatch at n={n}, index={i}").into());
            }
        }
        println!("saxpy n={n}: verified");
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact)))]
fn main() -> std::process::ExitCode {
    eprintln!("run with cargo metal run -p saxpy on macOS Apple Silicon");
    std::process::ExitCode::FAILURE
}

#[cfg(all(
    test,
    target_os = "macos",
    target_arch = "aarch64",
    metal_oxide_artifact
))]
mod tests {
    use metal_oxide::{Device, Error, LaunchConfig, Operation};

    #[test]
    #[ignore = "requires a Metal device and generated artifact"]
    fn generated_saxpy_matches_cpu() {
        super::main().unwrap();
    }

    #[test]
    #[ignore = "requires a Metal device and generated artifact"]
    fn unequal_lengths_violate_the_kernel_requirement() -> metal_oxide::Result<()> {
        let device = Device::system_default()?;
        let kernels = super::kernels::load(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
        let x = device.buffer_from_slice(&[1.0_f32, 2.0])?;
        let mut y = device.buffer_from_slice(&[1.0_f32; 3])?;
        let result = kernels
            .saxpy(LaunchConfig::<256>::for_elements(3)?, 2.0, &x, &mut y)
            .wait();
        assert!(matches!(
            result,
            Err(Error::Contract { relation, .. }) if relation == "x.len() == y.len()"
        ));
        assert_eq!(y.read().wait(), [1.0; 3]);
        Ok(())
    }
}
