use std::{path::Path, time::Instant};

use metal_oxide::Device;
use metal_oxide_artifact::Manifest;
use serde_json::json;

use crate::{
    Result,
    fgn::{self, Plan, Timing, elapsed},
    native,
    validation::compare,
};

pub fn run(path: &Path, samples: usize, warmup: usize, max_rows: u32) -> Result<()> {
    if samples == 0 || warmup == 0 || max_rows == 0 {
        return Err("samples, warmup and max rows must be positive".into());
    }
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
    let budget = (device.recommended_max_working_set_size() / 8).min(512 * 1024 * 1024);
    let mut cases = Vec::new();
    for size in [4096, 8192, 16384, 32768, 65536] {
        for rows in [1, 16, 256, 16384]
            .into_iter()
            .filter(|&rows| rows <= max_rows)
        {
            let capacity = (budget / (24 * size as u64)).max(1).min(rows as u64) as u32;
            let chunk = 1 << (31 - capacity.leading_zeros());
            let plan = Plan::new(size, chunk, 0, 0.7, 123, max_threads)?;
            let mut buffers = fgn::Buffers::new(&device, plan, 0.7)?;
            let mut baseline = native::Buffers::from(fgn::Buffers::new(&device, plan, 0.7)?);
            generated.run(&device, plan, &mut buffers)?;
            reference.run(plan, &mut baseline)?;
            compare(
                buffers.output.as_slice(),
                baseline.output(),
                plan.scale,
                "benchmark parity",
            );
            let mut rust_samples = Vec::new();
            let mut native_samples = Vec::new();
            let repetitions = if rows <= 16 { 16 } else { 1 };
            for iteration in 0..warmup + samples {
                let (rust, metal) = if iteration % 2 == 0 {
                    let rust = logical_run(plan, rows, repetitions, |p| {
                        generated.run(&device, p, &mut buffers)
                    })?;
                    let metal =
                        logical_run(plan, rows, repetitions, |p| reference.run(p, &mut baseline))?;
                    (rust, metal)
                } else {
                    let metal =
                        logical_run(plan, rows, repetitions, |p| reference.run(p, &mut baseline))?;
                    let rust = logical_run(plan, rows, repetitions, |p| {
                        generated.run(&device, p, &mut buffers)
                    })?;
                    (rust, metal)
                };
                compare(
                    buffers.output.as_slice(),
                    baseline.output(),
                    plan.scale,
                    "sample parity",
                );
                if iteration >= warmup {
                    rust_samples.push(rust);
                    native_samples.push(metal);
                }
            }
            let ratios = rust_samples
                .iter()
                .zip(&native_samples)
                .map(|(r, n)| r.total_ns as f64 / n.total_ns as f64)
                .collect::<Vec<_>>();
            let ratio = quantile(ratios.clone(), 0.5);
            println!("fGN {size} x {rows}, chunk {chunk}: Rust/native {ratio:.4}");
            cases.push(json!({
                "size": size, "rows": rows, "chunk_rows": chunk, "chunks": rows / chunk,
                "repetitions": repetitions,
                "generated_setup_ns": buffers.setup_ns, "native_setup_ns": baseline.setup_ns,
                "generated_allocation_upload_ns": buffers.allocation_ns,
                "native_allocation_upload_ns": baseline.allocation_ns,
                "generated": rust_samples, "native": native_samples,
                "total_ratio_p10": quantile(ratios.clone(), 0.1),
                "total_ratio_median": ratio, "total_ratio_p90": quantile(ratios, 0.9),
                "within_five_percent": ratio <= 1.05,
            }));
        }
    }
    let report = json!({
        "device": device.name(), "backend": std::env::var("METAL_OXIDE_BACKEND")?,
        "math_mode": manifest.abi.math_mode, "artifact": directory,
        "samples": samples, "warmup": warmup, "pair_order": "alternating",
        "native_backend": "classic MTL, setBytes, one encoder per pass",
        "working_set_budget_bytes": budget,
        "generated_load_ns": generated.load_ns, "native_load_ns": reference.load_ns,
        "generated_pipeline_ns": generated.pipeline_ns, "native_pipeline_ns": reference.pipeline_ns,
        "cases": cases,
    });
    std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

fn logical_run(
    plan: Plan,
    rows: u32,
    repetitions: usize,
    mut run: impl FnMut(Plan) -> Result<Timing>,
) -> Result<Timing> {
    let start = Instant::now();
    let mut total = Timing {
        gpu_ns: Some(0),
        ..Timing::default()
    };
    for first in (0..repetitions).flat_map(|_| (0..rows).step_by(plan.rows as usize)) {
        let current = Plan {
            base_cell: first as u64 * plan.n as u64,
            ..plan
        };
        let timing = run(current)?;
        total.encode_ns += timing.encode_ns;
        total.submit_ns += timing.submit_ns;
        total.wait_ns += timing.wait_ns;
        total.readback_ns += timing.readback_ns;
        total.gpu_ns = total.gpu_ns.zip(timing.gpu_ns).map(|(a, b)| a + b);
    }
    total.total_ns = elapsed(start);
    Ok(total)
}

fn quantile(mut values: Vec<f64>, fraction: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * fraction).round() as usize]
}
