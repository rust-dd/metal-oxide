use metal_oxide::device::{DisjointBuffer, ReadBuffer, thread_index};

/// Adds two buffers of equal length element by element.
#[metal_oxide::kernel(requires(a.len() == out.len(), b.len() == out.len()))]
pub fn vec_add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    if let Some(out) = out.get_mut(&index) {
        let i = index.get();
        *out = a.as_slice()[i] + b.as_slice()[i];
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use metal_oxide::{Device, LaunchConfig, Operation};

    let device = Device::system_default()?;
    println!("Device: {}", device.name());
    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        let a = (0..n).map(|i| (i % 1024) as f32 * 0.25).collect::<Vec<_>>();
        let b = (0..n).map(|i| (i % 511) as f32 * -0.5).collect::<Vec<_>>();
        let input_a = device.buffer_from_slice(&a)?;
        let input_b = device.buffer_from_slice(&b)?;
        let mut output = device.buffer_zeroed::<f32>(n as usize)?;
        let config = LaunchConfig::<256>::for_elements(n)?;
        vec_add(&device, config, &input_a, &input_b, &mut output).wait()?;
        for (i, actual) in output.read().wait().iter().enumerate() {
            if *actual != a[i] + b[i] {
                return Err(format!("vec_add mismatch at n={n}, index={i}").into());
            }
        }
        println!(
            "vec_add n={n}: verified ({} blocks x {} threads)",
            config.grid.x,
            LaunchConfig::<256>::BLOCK.x
        );
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("vec-add runs on macOS with Apple Silicon");
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests {
    use metal_oxide::{Device, Error, LaunchConfig, Operation};

    #[test]
    #[ignore = "requires a Metal device"]
    fn vec_add_matches_cpu() {
        super::main().unwrap();
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn unequal_lengths_violate_the_kernel_requirements() -> metal_oxide::Result<()> {
        let device = Device::system_default()?;
        let a = device.buffer_from_slice(&[1.0_f32, 2.0])?;
        let mut out = device.buffer_zeroed::<f32>(3)?;
        let config = LaunchConfig::<256>::for_elements(3)?;
        let result = super::vec_add(&device, config, &a, &a, &mut out).wait();
        assert!(matches!(
            result,
            Err(Error::Contract { relation, .. }) if relation == "a.len() == out.len()"
        ));
        assert_eq!(out.read().wait(), [0.0; 3]);
        Ok(())
    }
}
