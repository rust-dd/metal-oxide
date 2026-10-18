mod gemm;
mod matrix;
mod particle;
mod scalar;
mod scan;
mod sort;

use std::{
    collections::{BTreeMap, BTreeSet},
    future::{Future, IntoFuture},
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    time::Instant,
};

use metal_oxide::{Batch, Device, Launch, Module, Operation, Pipeline, sequence};
use serde::{Deserialize, Serialize};

use crate::stats::Summary;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The workloads a configuration selects from, in the order each variant runs them; gemm's
/// candidates then take turns sample by sample.
const WORKLOADS: [&str; 7] = [
    "vec-add",
    "reduction",
    "cooperation",
    "particle-update",
    "matmul",
    "sort",
    "gemm",
];

/// The most keys a sort case sorts.
const SORT_KEYS: u32 = 1 << 24;

#[derive(Deserialize)]
struct Config {
    artifacts: BTreeMap<String, PathBuf>,
    references: BTreeMap<String, PathBuf>,
    /// The selected workloads.
    cases: Vec<String>,
    samples: usize,
    warmup: usize,
    elements: u32,
    matrix: [u32; 3],
    gemm: [u32; 3],
    gemm_repeat: u32,
    sort: u32,
}

impl Config {
    fn validate(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        let mut cases = self.cases.iter();
        let distinct = cases.all(|case| WORKLOADS.contains(&case.as_str()) && seen.insert(case));
        if self.cases.is_empty() || !distinct {
            let workloads = WORKLOADS.join(", ");
            return Err(format!("cases must be distinct workloads of {workloads}").into());
        }
        if self.samples == 0
            || self.elements == 0
            || self.elements > 16_000_000
            || self.matrix.iter().any(|&n| n == 0 || n > 1024)
            || self.gemm.iter().any(|&n| n == 0 || n > 8192)
        {
            return Err(
                "samples, elements, matrix and gemm dimensions must be positive and bounded".into(),
            );
        }
        if !(1..=256).contains(&self.gemm_repeat) {
            return Err("gemm repeat must be from 1 to 256".into());
        }
        if !(1..=SORT_KEYS).contains(&self.sort) {
            return Err(format!("sort must be from 1 to {SORT_KEYS} keys").into());
        }
        Ok(())
    }

    fn selects(&self, workload: &str) -> bool {
        self.cases.iter().any(|case| case == workload)
    }
}

#[derive(Default, Serialize)]
struct Sample {
    cpu_reference_ns: Option<u64>,
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
    /// Identical runs per sample, which share one command buffer.
    repeat: u32,
    /// Threads per threadgroup of our kernels; absent when a library picks its own.
    block: Option<[u32; 3]>,
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
            repeat: 1,
            block: Some(match name {
                "matmul" => [16, 16, 1],
                "particle-update" => [64, 1, 1],
                "gemm-f32" | "gemm-f16" => [128, 1, 1],
                _ => [256, 1, 1],
            }),
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
                ("cpu_reference_ns", sample.cpu_reference_ns),
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

/// Runs the operations `encode` pushes as one command buffer and times its phases.
fn dispatch(encode: impl FnOnce(&mut Batch) -> metal_oxide::Result<()>) -> Result<Sample> {
    let mut sample = Sample::default();
    let start = Instant::now();
    let operation = sequence(|batch| {
        let start = Instant::now();
        let result = encode(batch);
        sample.encode_ns = elapsed(start);
        result
    });
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = operation.timed().into_future();
    let mut poll = Pin::new(&mut future).poll(&mut cx);
    sample.submit_ns = elapsed(start);
    let start = Instant::now();
    let ((), gpu) = loop {
        match poll {
            Poll::Ready(result) => break result?,
            Poll::Pending => {
                std::thread::park();
                poll = Pin::new(&mut future).poll(&mut cx);
            }
        }
    };
    sample.wait_ns = elapsed(start);
    sample.gpu_ns = gpu.map(|d| d.as_nanos().try_into().unwrap());
    Ok(sample)
}

struct Unpark(std::thread::Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Whether this process also measures MPS, which ignores the runtime's backend: only the classic
/// run does.
fn measures_mps(backend: &str) -> bool {
    backend == "classic"
}

/// One report per measured backend, with MPS's cases as a backend of their own.
fn backend_reports(
    device: &str,
    backend: &str,
    cases: Vec<Case>,
    mps: Option<Vec<Case>>,
) -> Vec<serde_json::Value> {
    let mut backends =
        vec![serde_json::json!({"device": device, "backend": backend, "cases": cases})];
    if let Some(mps) = mps {
        backends.push(serde_json::json!({"device": device, "backend": "mps", "cases": mps}));
    }
    backends
}

pub(super) fn run() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let config: Config =
        serde_json::from_slice(&std::fs::read(args.next().ok_or("missing config")?)?)?;
    let output = args.next().ok_or("missing output path")?;
    config.validate()?;
    let backend = std::env::var("METAL_OXIDE_BACKEND")?;
    let device = Device::system_default()?;
    let mut cases = Vec::new();
    for variant in ["rust", "msl"] {
        for workload in WORKLOADS {
            if config.selects(workload) {
                cases.extend(measure(&device, &config, workload, variant)?);
            }
        }
    }
    let mut mps = None;
    if config.selects("gemm") {
        let problem = gemm::Problem::new(config.gemm);
        let (gemm, measured) = gemm::run(&device, &config, &problem, measures_mps(&backend))?;
        cases.extend(gemm);
        mps = measured;
    }
    let backends = backend_reports(&device.name(), &backend, cases, mps);
    std::fs::write(output, serde_json::to_vec_pretty(&backends)?)?;
    Ok(())
}

/// The cases of `workload` for `variant`; gemm runs after both variants instead.
fn measure(device: &Device, config: &Config, workload: &str, variant: &str) -> Result<Vec<Case>> {
    Ok(match workload {
        "vec-add" => vec![scalar::floats(device, config, variant, false)?],
        "reduction" => vec![scalar::floats(device, config, variant, true)?],
        "cooperation" => vec![
            scalar::histogram(device, config, variant)?,
            scan::run(device, config, variant)?,
        ],
        "particle-update" => vec![particle::run(device, config, variant)?],
        "matmul" => vec![matrix::run(device, config, variant)?],
        "sort" => vec![sort::run(device, config, variant)?],
        "gemm" => Vec::new(),
        _ => unreachable!("{workload} is not a workload"),
    })
}

#[cfg(test)]
mod tests;
