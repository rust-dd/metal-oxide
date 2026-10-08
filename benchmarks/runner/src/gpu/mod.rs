mod matrix;
mod particle;
mod scalar;
mod scan;

use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use metal_oxide::{Batch, Device, Module, Pipeline};
use serde::{Deserialize, Serialize};

use crate::stats::Summary;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[allow(dead_code)]
mod records {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[derive(Deserialize)]
struct Config {
    artifacts: BTreeMap<String, PathBuf>,
    references: BTreeMap<String, PathBuf>,
    samples: usize,
    warmup: usize,
    elements: u32,
    matrix: [u32; 3],
}

#[derive(Default, Serialize)]
struct Sample {
    cpu_reference_ns: u64,
    input_copy_ns: u64,
    upload_ns: u64,
    encode_ns: u64,
    submit_ns: u64,
    wait_ns: u64,
    gpu_ns: Option<u64>,
    readback_ns: u64,
}

#[derive(Serialize)]
struct Case {
    name: String,
    variant: String,
    shape: Vec<u32>,
    block: [u32; 3],
    library_load_ns: u64,
    pipeline_ns: u64,
    allocation_ns: u64,
    samples: Vec<Sample>,
    summary_ns: BTreeMap<String, Summary>,
}

impl Case {
    fn new(name: &str, variant: &str, shape: Vec<u32>) -> Self {
        Self {
            name: name.into(),
            variant: variant.into(),
            shape,
            block: match name {
                "matmul" => [16, 16, 1],
                "particle-update" => [64, 1, 1],
                _ => [256, 1, 1],
            },
            library_load_ns: 0,
            pipeline_ns: 0,
            allocation_ns: 0,
            samples: vec![],
            summary_ns: BTreeMap::new(),
        }
    }

    fn pipelines(
        &mut self,
        device: &Device,
        config: &Config,
        package: &str,
        entries: &[&str],
    ) -> Result<Vec<Pipeline>> {
        let start = Instant::now();
        let module = if self.variant == "rust" {
            Module::from_artifact(device, &config.artifacts[package])?
        } else {
            Module::from_metallib(device, &config.references[package])?
        };
        self.library_load_ns = elapsed(start);
        let start = Instant::now();
        let pipelines = entries
            .iter()
            .map(|entry| Pipeline::new(device, &module, entry))
            .collect::<metal_oxide::Result<Vec<_>>>()?;
        self.pipeline_ns = elapsed(start);
        Ok(pipelines)
    }

    fn finish(mut self) -> Result<Self> {
        if self.samples.is_empty() {
            return Err("cannot report a workload without measured samples".into());
        }
        let mut measurements = BTreeMap::<_, Vec<u64>>::new();
        for sample in &self.samples {
            for (name, value) in [
                ("cpu_reference_ns", Some(sample.cpu_reference_ns)),
                ("input_copy_ns", Some(sample.input_copy_ns)),
                ("upload_ns", Some(sample.upload_ns)),
                ("encode_ns", Some(sample.encode_ns)),
                ("submit_ns", Some(sample.submit_ns)),
                ("wait_ns", Some(sample.wait_ns)),
                ("gpu_ns", sample.gpu_ns),
                ("readback_ns", Some(sample.readback_ns)),
            ] {
                if let Some(value) = value {
                    measurements.entry(name).or_default().push(value);
                }
            }
        }
        for (name, values) in measurements {
            self.summary_ns.insert(name.into(), Summary::new(&values)?);
        }
        Ok(self)
    }
}

fn elapsed(start: Instant) -> u64 {
    start
        .elapsed()
        .as_nanos()
        .try_into()
        .expect("benchmark interval fits u64 nanoseconds")
}

// SAFETY: the caller supplies valid, disjoint resources and collective launch shapes.
unsafe fn dispatch(
    device: &Device,
    encode: impl FnOnce(&mut Batch<'_>) -> metal_oxide::Result<()>,
) -> Result<Sample> {
    let mut sample = Sample::default();
    let start = Instant::now();
    // SAFETY: the workload establishes each kernel's bounds, aliasing and participation contract.
    let submission = unsafe {
        device.submit(|batch| {
            let start = Instant::now();
            let result = encode(batch);
            sample.encode_ns = elapsed(start);
            result
        })?
    };
    let submitted = elapsed(start);
    let start = Instant::now();
    let report = submission.wait()?;
    sample.wait_ns = elapsed(start);
    sample.submit_ns = submitted;
    sample.gpu_ns = report
        .gpu_duration
        .map(|d| d.as_nanos().try_into().unwrap());
    Ok(sample)
}

pub(super) fn run() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let config: Config =
        serde_json::from_slice(&std::fs::read(args.next().ok_or("missing config")?)?)?;
    let output = args.next().ok_or("missing output path")?;
    if config.samples == 0
        || config.elements == 0
        || config.elements > 16_000_000
        || config.matrix.iter().any(|&n| n == 0 || n > 1024)
    {
        return Err("samples, elements and matrix dimensions must be positive and bounded".into());
    }
    let device = Device::system_default()?;
    let mut cases = Vec::new();
    for variant in ["rust", "msl"] {
        cases.extend(scalar::run(&device, &config, variant)?);
        cases.push(scan::run(&device, &config, variant)?);
        cases.push(particle::run(&device, &config, variant)?);
        cases.push(matrix::run(&device, &config, variant)?);
    }
    let report = serde_json::json!({"device": device.name(), "backend": std::env::var("METAL_OXIDE_BACKEND")?, "cases": cases});
    std::fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_requires_samples_and_keeps_unavailable_gpu_time_absent() {
        assert!(Case::new("vec-add", "rust", vec![1]).finish().is_err());
        let mut case = Case::new("vec-add", "rust", vec![1]);
        case.samples.push(Sample {
            submit_ns: 100,
            ..Sample::default()
        });
        let report = case.finish().unwrap();
        assert!(!report.summary_ns.contains_key("gpu_ns"));
        assert_eq!(report.summary_ns["submit_ns"].median, 100);
        assert!(serde_json::to_value(&report).unwrap()["samples"][0]["gpu_ns"].is_null());
    }
}
