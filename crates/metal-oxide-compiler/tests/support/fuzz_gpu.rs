use super::support::{
    self,
    differential::{Case, Outcome, Run},
    oracle::Failure,
};
use std::{path::Path, process::Command, time::Duration};

const SOURCE: &str = "METAL_OXIDE_FUZZ_GPU_SOURCE";
const GENERATED: &str = "METAL_OXIDE_FUZZ_GPU_GENERATED";
const RESULT: &str = "METAL_OXIDE_FUZZ_GPU_RESULT";
const TEST: &str = "generated_programs_agree_until_the_time_budget_ends";

type Output = Result<(Option<(u32, u32)>, Vec<u32>, Vec<u32>), String>;

/// Runs one GPU oracle in this process when the parent supplies its case.
pub fn child() -> bool {
    let Some(source) = std::env::var_os(SOURCE) else {
        return false;
    };
    let generated = std::env::var_os(GENERATED).unwrap();
    let result = std::env::var_os(RESULT).unwrap();
    let case = Case::saved(Path::new(&source));
    let output: Output = metal_oxide::Device::system_default()
        .map_err(|error| error.to_string())
        .and_then(|device| case.gpu(&device, Path::new(&generated)))
        .map(|gpu| {
            let failure = gpu.failure.map(|failure| match failure {
                Outcome::Panic { line, column } => (line, column),
                Outcome::Words(_) => unreachable!("a GPU failure is a runtime check"),
            });
            (failure, gpu.words, gpu.counters)
        });
    std::fs::write(result, serde_json::to_vec(&output).unwrap()).unwrap();
    true
}

/// A failed Metal compiler can poison later pipelines, so each candidate gets a process.
pub fn check(case: &Case, expected: &Run, generated: &Path) -> Result<(), Failure> {
    let result = case.directory.join("gpu-result.json");
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", TEST, "--ignored", "--nocapture"])
        .env(SOURCE, &case.source)
        .env(GENERATED, generated)
        .env(RESULT, &result);
    let output = support::process::output(&mut command, Duration::from_secs(120))
        .map_err(|error| Failure::Device(format!("GPU process: {error}")))?;
    support::succeeded(output).map_err(Failure::Device)?;
    let output =
        std::fs::read(&result).map_err(|error| Failure::Device(format!("GPU result: {error}")))?;
    let output = serde_json::from_slice::<Output>(&output)
        .map_err(|error| Failure::Device(format!("GPU result: {error}")))?;
    let (failure, words, counters) = output.map_err(Failure::Device)?;
    let failure = failure.map(|(line, column)| Outcome::Panic { line, column });
    match case.gpu_difference(expected, failure, &words, &counters) {
        Some(mismatch) => Err(Failure::Gpu(mismatch)),
        None => Ok(()),
    }
}
