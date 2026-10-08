mod kernels {
    include!(env!("METAL_OXIDE_BINDINGS"));
}
mod scan;

use kernels::{Config, Particle};
use metal_oxide::{Device, LaunchConfig};
use std::path::Path;

fn verify(artifact: &Path, n: u32) -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let kernels = kernels::load(&device, artifact)?;
    let initial = (0..n)
        .map(|i| Particle {
            position: ((i % 16) as f32, (i % 7) as f32),
            velocity: [1.0, 2.0],
            tag: (i % 3) as u8,
        })
        .collect::<Vec<_>>();
    let expected = initial
        .iter()
        .map(|value| {
            let mut value = *value;
            if value.tag != 0 {
                value.position.0 += 2.0;
                value.position.1 += 4.0;
            }
            value
        })
        .collect::<Vec<_>>();
    let flags = expected
        .iter()
        .map(|value| {
            let x = value.position.0;
            x.mul_add(x, 0.0).sqrt() as u32 & 1
        })
        .collect::<Vec<_>>();
    let mut sum = 0_u32;
    let prefix = flags
        .iter()
        .map(|value| {
            let previous = sum;
            sum += value;
            previous
        })
        .collect::<Vec<_>>();
    let selected = expected
        .iter()
        .zip(&flags)
        .filter_map(|(value, flag)| (*flag != 0).then_some(*value))
        .collect::<Vec<_>>();
    let input = device.buffer_from_slice(&initial)?;
    let mut updated = device.buffer_zeroed::<Particle>(n as usize)?;
    let mut gpu_flags = device.buffer_zeroed::<u32>(n as usize)?;
    let mut compact = device.buffer_zeroed::<Particle>(n as usize)?;
    let mut scan = scan::Scan::new(&device, n)?;
    let launch = LaunchConfig::<64>::for_elements(n)?;
    // SAFETY: distinct buffers cover n; ordered passes establish flags and their prefix before
    // scatter. Complete 64-thread blocks participate in every cooperative scan operation.
    unsafe {
        device.submit(|batch| {
            kernels.enqueue_update(batch, launch, &input, &mut updated, Config { dt: 1.0 }, n)?;
            kernels.enqueue_select(batch, launch, &updated, &mut gpu_flags, n)?;
            scan.enqueue(batch, &kernels, &gpu_flags)?;
            kernels.enqueue_scatter(
                batch,
                launch,
                &updated,
                &gpu_flags,
                scan.output(),
                &mut compact,
                n,
            )
        })?
    }
    .wait()?;
    assert_eq!(input.as_slice(), initial);
    assert_eq!(updated.as_slice(), expected);
    assert_eq!(gpu_flags.as_slice(), flags);
    assert_eq!(scan.output().as_slice(), prefix);
    assert_eq!(&compact.as_slice()[..selected.len()], selected);
    println!(
        "{}: n={n}, selected={sum}, all passes match CPU",
        device.name()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut n = 257;
    let mut artifact = env!("METAL_OXIDE_ARTIFACT_DIR").to_owned();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--n" => n = args.next().ok_or("--n requires a value")?.parse()?,
            "--artifact" => artifact = args.next().ok_or("--artifact requires a path")?,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    verify(Path::new(&artifact), n)?;
    Ok(())
}

#[test]
#[ignore = "requires the installed artifact and Metal hardware"]
fn installed_chain() {
    verify(Path::new(env!("METAL_OXIDE_ARTIFACT_DIR")), 257).unwrap();
}
