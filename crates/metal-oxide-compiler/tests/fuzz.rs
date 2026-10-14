#![cfg(feature = "rustc-private")]

mod support;

use std::{
    path::Path,
    time::{Duration, Instant, SystemTime},
};
use support::{differential::Case, generate::Program, oracle::Failure};

/// The oracle on the host, and on the GPU that `METAL_OXIDE_FUZZ_GPU` asks for.
struct Oracle {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    device: Option<metal_oxide::Device>,
}

impl Oracle {
    fn new() -> Self {
        Self {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            device: std::env::var_os("METAL_OXIDE_FUZZ_GPU")
                .map(|_| metal_oxide::Device::system_default().unwrap()),
        }
    }

    fn check(&self, case: &Case) -> Result<(), Failure> {
        let (_expected, _generated) = case.check()?;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some(device) = &self.device {
            case.check_gpu(device, &_expected, &_generated)?;
        }
        Ok(())
    }
}

fn variable(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Writes the program, its minimized form and the failure under `directory`; minimizing stops
/// after `METAL_OXIDE_FUZZ_SHRINK_SECONDS`.
fn report(oracle: &Oracle, seed: u64, program: Program, failure: &Failure, directory: &Path) {
    std::fs::create_dir_all(directory).unwrap();
    std::fs::write(directory.join("failure.txt"), failure.to_string()).unwrap();
    let original = Case::new(&format!("{seed:016x}"), &program, seed);
    std::fs::copy(&original.source, directory.join("case.rs")).unwrap();
    let seconds = variable("METAL_OXIDE_FUZZ_SHRINK_SECONDS").unwrap_or(600);
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let shrunk = program.shrink(|candidate| {
        let case = Case::new(&format!("{seed:016x}-shrink"), candidate, seed);
        Instant::now() < deadline
            && oracle
                .check(&case)
                .is_err_and(|other| other.same_as(failure))
    });
    let minimized = Case::new(&format!("{seed:016x}-minimized"), &shrunk, seed);
    std::fs::copy(&minimized.source, directory.join("minimized.rs")).unwrap();
}

/// Checks seeds from `METAL_OXIDE_FUZZ_SEED` (by default the clock) until
/// `METAL_OXIDE_FUZZ_SECONDS` elapse, and writes each failing program, minimized, under
/// `target/fuzz/<seed>/`.
#[test]
#[ignore = "fuzzes until METAL_OXIDE_FUZZ_SECONDS elapse"]
fn generated_programs_agree_until_the_time_budget_ends() {
    let seconds = variable("METAL_OXIDE_FUZZ_SECONDS").unwrap_or(60);
    let mut seed = variable("METAL_OXIDE_FUZZ_SEED").unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });
    let oracle = Oracle::new();
    let output = support::root().join("target/fuzz");
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut checked = 0;
    let mut failures = Vec::new();
    while Instant::now() < deadline {
        let program = Program::generate(seed);
        let case = Case::new(&format!("{seed:016x}"), &program, seed);
        if let Err(failure) = oracle.check(&case) {
            eprintln!("seed {seed:#x} fails: {failure}");
            report(
                &oracle,
                seed,
                program,
                &failure,
                &output.join(format!("{seed:016x}")),
            );
            failures.push(seed);
        }
        checked += 1;
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    }
    eprintln!("checked {checked} programs");
    assert!(
        failures.is_empty(),
        "failing seeds: {failures:#x?}, minimized under {}",
        output.display()
    );
}
