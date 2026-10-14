#![cfg(feature = "rustc-private")]

mod support;

use std::time::{Duration, Instant, SystemTime};

/// The outcome of one seed, or the reason it fails.
fn check(seed: u64) -> Result<(), String> {
    let case = support::differential::Case::generated(seed);
    let result = std::panic::catch_unwind(|| {
        let generated = case.compile();
        case.mismatch(&case.cpu(), &case.msl(&generated))
    });
    match result {
        Ok(None) => Ok(()),
        Ok(Some(mismatch)) => Err(format!("host MSL differs: {mismatch}")),
        Err(panic) => Err(panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|text| (*text).to_owned()))
            .unwrap_or_default()),
    }
}

/// Checks seeds from `METAL_OXIDE_FUZZ_SEED` (by default the clock) until
/// `METAL_OXIDE_FUZZ_SECONDS` elapse, and lists the seeds that failed.
#[test]
#[ignore = "fuzzes until METAL_OXIDE_FUZZ_SECONDS elapse"]
fn generated_programs_agree_until_the_time_budget_ends() {
    let variable = |name: &str| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
    };
    let seconds = variable("METAL_OXIDE_FUZZ_SECONDS").unwrap_or(60);
    let mut seed = variable("METAL_OXIDE_FUZZ_SEED").unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut checked = 0;
    let mut failures = Vec::new();
    while Instant::now() < deadline {
        if let Err(reason) = check(seed) {
            eprintln!("seed {seed:#x} fails: {reason}");
            failures.push(seed);
        }
        checked += 1;
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    }
    eprintln!("checked {checked} programs");
    assert!(failures.is_empty(), "failing seeds: {failures:#x?}");
}
