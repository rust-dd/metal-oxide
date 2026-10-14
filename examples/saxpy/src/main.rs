use metal_oxide::device::{DisjointBuffer, ReadBuffer, thread_index};

/// `y = a * x + y`: each thread reads and writes its own element of `y`.
#[metal_oxide::kernel(requires(x.len() == y.len()))]
pub fn saxpy(a: f32, x: ReadBuffer<f32>, mut y: DisjointBuffer<f32>) {
    let index = thread_index();
    if let Some(y) = y.get_mut(&index) {
        *y += a * x.as_slice()[index.get()];
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use metal_oxide::{Device, LaunchConfig, Operation};

    let device = Device::system_default()?;
    println!("Device: {}", device.name());
    let a = 2.5_f32;
    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        let x = (0..n).map(|i| (i % 1024) as f32 * 0.25).collect::<Vec<_>>();
        let y = (0..n).map(|i| (i % 511) as f32 * -0.5).collect::<Vec<_>>();
        let input = device.buffer_from_slice(&x)?;
        let mut output = device.buffer_from_slice(&y)?;
        let config = LaunchConfig::<256>::for_elements(n)?;
        saxpy(&device, config, a, &input, &mut output).wait()?;
        for (i, actual) in output.read().wait().iter().enumerate() {
            if *actual != a * x[i] + y[i] {
                return Err(format!("saxpy mismatch at n={n}, index={i}").into());
            }
        }
        println!("saxpy n={n}: verified");
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("saxpy runs on macOS with Apple Silicon");
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests {
    use metal_oxide::{Device, Error, LaunchConfig, Operation};

    #[test]
    #[ignore = "requires a Metal device"]
    fn saxpy_matches_cpu() {
        super::main().unwrap();
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn unequal_lengths_violate_the_kernel_requirement() -> metal_oxide::Result<()> {
        let device = Device::system_default()?;
        let x = device.buffer_from_slice(&[1.0_f32, 2.0])?;
        let mut y = device.buffer_from_slice(&[1.0_f32; 3])?;
        let config = LaunchConfig::<256>::for_elements(3)?;
        let result = super::saxpy(&device, config, 2.0, &x, &mut y).wait();
        assert!(matches!(
            result,
            Err(Error::Contract { relation, .. }) if relation == "x.len() == y.len()"
        ));
        assert_eq!(y.read().wait(), [1.0; 3]);
        Ok(())
    }
}
