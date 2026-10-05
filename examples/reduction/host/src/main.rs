#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod kernels {
    include!(env!("METAL_OXIDE_BINDINGS"));
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use metal_oxide::{Device, LaunchConfig};
    let device = Device::system_default()?;
    let kernels = kernels::load(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        let values = (0..n).map(|i| (i % 17) as f32 - 8.0).collect::<Vec<_>>();
        let input = device.buffer_from_slice(&values)?;
        let mut output = device.buffer_zeroed::<f32>(n.div_ceil(256) as usize)?;
        // SAFETY: each 256-thread block owns one shared tile and output; input covers n elements.
        unsafe {
            kernels.reduce(
                LaunchConfig::<256>::for_elements(n)?,
                &input,
                &mut output,
                n,
            )?;
        }
        let expected = values
            .chunks(256)
            .map(|v| v.iter().sum::<f32>())
            .collect::<Vec<_>>();
        assert_eq!(output.as_slice(), expected, "n={n}");
        println!("reduction n={n}: verified on {}", device.name());
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact)))]
fn main() -> std::process::ExitCode {
    eprintln!("run with cargo metal run -p reduction on macOS Apple Silicon");
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
fn generated_reduction_matches_cpu() {
    main().unwrap();
}
