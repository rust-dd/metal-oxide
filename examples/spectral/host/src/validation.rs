use metal_oxide::{Buffer, Device, LaunchConfig};
use metal_oxide_artifact::Manifest;

use crate::{
    Result, cpu,
    fgn::{self, Plan},
    kernels, native,
};

pub fn compare(actual: &[f32], expected: &[f32], scale: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 4e-5 * scale.max(expected.abs());
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{label}[{i}]: {actual} != {expected}, tolerance {tolerance}"
        );
    }
}

pub fn run() -> Result<()> {
    let device = Device::system_default()?;
    let directory = env!("METAL_OXIDE_ARTIFACT_DIR");
    let manifest = Manifest::from_json(&std::fs::read_to_string(format!(
        "{directory}/manifest.json"
    ))?)?;
    let generated = fgn::Context::new(&device, directory)?;
    let reference = native::Context::new(&device, manifest.abi.math_mode)?;
    let max_threads = generated
        .max_threads()
        .min(reference.max_threads())
        .min(device.max_threadgroup_memory_per_block() / 16);
    println!(
        "{}: {:?}; load {} / {} ns, pipelines {} / {} ns",
        device.name(),
        manifest.abi.math_mode,
        generated.load_ns,
        reference.load_ns,
        generated.pipeline_ns,
        reference.pipeline_ns
    );
    for size in [0, 1, 2, 3, 7, 32, 257, 4096, 16384, 65536] {
        for (rows, first) in [(0, 0), (1, 0), (3, 1), (7, u32::MAX as u64 + 5)] {
            let plan = Plan::new(size, rows, first, 0.7, 123, max_threads)?;
            let mut buffers = fgn::Buffers::new(&device, plan, 0.7)?;
            let mut baseline = native::Buffers::from(fgn::Buffers::new(&device, plan, 0.7)?);
            let cpu = cpu::fgn(
                buffers.roots.as_slice(),
                size as usize,
                rows as usize,
                first,
                plan.seed,
                plan.scale,
            );
            generated.run(&device, plan, &mut buffers)?;
            reference.run(plan, &mut baseline)?;
            compare(buffers.output.as_slice(), &cpu, plan.scale, "CPU");
            compare(
                buffers.output.as_slice(),
                baseline.output(),
                plan.scale,
                "native",
            );
        }
        println!("fGN n={size}: CPU and native parity");
    }
    chunking(&device, &generated, max_threads)?;
    law(&device, &generated, max_threads)?;
    interop(&device, &reference, max_threads, directory)?;
    crate::sheet::run(&device, directory, manifest.abi.math_mode)?;
    crate::euler::run(&device, directory, manifest.abi.math_mode)?;
    Ok(())
}

fn law(device: &Device, context: &fgn::Context, max_threads: usize) -> Result<()> {
    for hurst in [0.3_f64, 0.5, 0.7] {
        let plan = Plan::new(257, 512, 0, hurst, u32::MAX, max_threads)?;
        let mut buffers = fgn::Buffers::new(device, plan, hurst)?;
        context.run(device, plan, &mut buffers)?;
        let values = buffers.output.as_slice();
        let mean = values
            .iter()
            .map(|&x| x as f64 / plan.scale as f64)
            .sum::<f64>()
            / values.len() as f64;
        let variance = values
            .iter()
            .map(|&x| (x as f64 / plan.scale as f64).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        let covariance = values
            .chunks(plan.size as usize)
            .flat_map(|row| row.windows(2))
            .map(|pair| pair[0] as f64 * pair[1] as f64 / (plan.scale as f64).powi(2))
            .sum::<f64>()
            / (plan.rows * (plan.size - 1)) as f64;
        let expected = 0.5 * (2.0_f64.powf(2.0 * hurst) - 2.0);
        assert!(mean.abs() < 0.04, "H={hurst}, mean={mean}");
        assert!(
            (variance - 1.0).abs() < 0.05,
            "H={hurst}, variance={variance}"
        );
        assert!(
            (covariance - expected).abs() < 0.04,
            "H={hurst}, covariance={covariance}, expected={expected}"
        );
    }
    Ok(())
}

fn chunking(device: &Device, context: &fgn::Context, max_threads: usize) -> Result<()> {
    for (size, first) in [(257, 0_u64), (4096, 1), (32, u64::MAX - 7)] {
        let plan = Plan::new(size, 9, first, 0.3, 0xfedc_ba98, max_threads)?;
        let mut buffers = fgn::Buffers::new(device, plan, 0.3)?;
        context.run(device, plan, &mut buffers)?;
        let full = buffers.output.as_slice().to_vec();
        let mut chunked = Vec::new();
        let mut offset = 0;
        for rows in [1, 3, 2, 3] {
            let plan = Plan::new(
                size,
                rows,
                first.wrapping_add(offset),
                0.3,
                0xfedc_ba98,
                max_threads,
            )?;
            let mut buffers = fgn::Buffers::new(device, plan, 0.3)?;
            context.run(device, plan, &mut buffers)?;
            chunked.extend_from_slice(buffers.output.as_slice());
            offset += rows as u64;
        }
        assert_eq!(full, chunked, "chunk invariance n={size}, first={first}");
    }
    Ok(())
}

fn interop(
    device: &Device,
    reference: &native::Context,
    max_threads: usize,
    directory: &str,
) -> Result<()> {
    let plan = Plan::new(257, 3, 1, 0.7, 987, max_threads)?;
    let mut native = native::Buffers::from(fgn::Buffers::new(device, plan, 0.7)?);
    reference.run(plan, &mut native)?;
    let expected = native
        .output()
        .chunks(plan.size as usize)
        .flat_map(|row| {
            let mut state = 1.0_f32;
            row.iter().map(move |increment| {
                state = state + 0.25 * state * 0.001 + increment;
                state
            })
        })
        .collect::<Vec<_>>();
    // SAFETY: the native producer completed; consuming the buffer set drops its
    // other owners and hands over initialized shared storage without live aliases.
    let mut values =
        unsafe { Buffer::<f32>::from_native(device, native.into_output(), expected.len())? };
    let kernels = kernels::load(device, directory)?;
    // SAFETY: each lane integrates one complete row of native fGN increments in place.
    unsafe {
        kernels.integrate(
            LaunchConfig::<32>::for_elements(plan.rows)?,
            &mut values,
            plan.rows,
            plan.size,
            0.001,
        )?;
    }
    compare(
        values.as_slice(),
        &expected,
        1.0,
        "native fGN -> generated Euler",
    );
    Ok(())
}
