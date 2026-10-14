//! An application outside the metal-oxide workspace whose kernels plain Cargo builds once
//! `cargo metal setup` has installed the toolchain.

use metal_oxide::device::{DisjointBuffer, ReadBuffer, thread_index};

/// Multiplies each element by `factor`.
#[metal_oxide::kernel(requires(input.len() == output.len()))]
pub fn scale(input: ReadBuffer<u32>, mut output: DisjointBuffer<u32>, factor: u32) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        *output = input.as_slice()[index.get()] * factor;
    }
}

/// Particles that the host creates and checks and the kernels move.
#[metal_oxide::kernels(dependencies(installed_helper))]
pub mod particles {
    use metal_oxide::device::{DisjointBuffer, thread_index};

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Particle {
        pub position: [f32; 2],
        pub velocity: [f32; 2],
    }

    /// Moves each particle `steps` steps of `dt`.
    #[kernel]
    pub fn advance(mut particles: DisjointBuffer<Particle>, dt: f32, steps: u32) {
        let index = thread_index();
        if let Some(particle) = particles.get_mut(&index) {
            let moved = installed_helper::advance(particle.position, particle.velocity, dt, steps);
            particle.position = moved;
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn run() -> metal_oxide::Result<()> {
    use metal_oxide::{Device, LaunchConfig, Operation};
    use particles::Particle;

    let device = Device::system_default()?;
    let input = (0..1000_u32).collect::<Vec<_>>();
    let values = device.buffer_from_slice(&input)?;
    let mut scaled = device.buffer_zeroed::<u32>(input.len())?;
    let config = LaunchConfig::<256>::for_elements(1000)?;
    scale(&device, config, &values, &mut scaled, 3).wait()?;
    let expected = input.iter().map(|value| value * 3).collect::<Vec<_>>();
    assert_eq!(scaled.read().wait(), expected);

    // Steps of exact binary fractions match whether or not the GPU fuses them.
    let initial = (0..1000)
        .map(|i| Particle {
            position: [i as f32, 0.0],
            velocity: [0.5, -0.25],
        })
        .collect::<Vec<_>>();
    let mut moved = device.buffer_from_slice(&initial)?;
    let config = LaunchConfig::<64>::for_elements(1000)?;
    particles::advance(&device, config, &mut moved, 0.125, 4).wait()?;
    let expected = initial
        .iter()
        .map(|particle| Particle {
            position: installed_helper::advance(particle.position, particle.velocity, 0.125, 4),
            ..*particle
        })
        .collect::<Vec<_>>();
    assert_eq!(moved.read().wait(), expected);
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> metal_oxide::Result<()> {
    run()?;
    println!("the kernels match the CPU");
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("this application runs on macOS with Apple Silicon");
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests {
    #[test]
    #[ignore = "requires a Metal device"]
    fn kernels_match_the_cpu() -> metal_oxide::Result<()> {
        super::run()
    }
}
