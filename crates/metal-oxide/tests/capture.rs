#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use metal_oxide::{Argument, Device, Dim3, Launch, LaunchConfig, Module, Operation, Pipeline};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

const CHILD: &str = "METAL_OXIDE_CAPTURE_CHILD";
const SOURCE: &str = "#include <metal_stdlib>\nusing namespace metal;\nkernel void captured_fill(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) { out[i] = i; }";

/// Metal reads `METAL_CAPTURE_ENABLED` at process start, so each case runs in its own process.
fn run(trace: &Path, enabled: bool) -> (bool, String) {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--ignored",
            "--exact",
            "the_first_device_records_a_trace_until_it_drops",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("METAL_OXIDE_CAPTURE", trace)
        .env_remove("METAL_CAPTURE_ENABLED");
    if enabled {
        command.env("METAL_CAPTURE_ENABLED", "1");
    }
    let output = command.output().unwrap();
    let text = String::from_utf8_lossy(&output.stdout) + String::from_utf8_lossy(&output.stderr);
    (output.status.success(), text.into_owned())
}

fn record(trace: &Path) -> metal_oxide::Result<()> {
    {
        let device = Device::system_default()?;
        let module = Module::from_source(&device, SOURCE)?;
        let pipeline = Pipeline::new(&device, &module, "captured_fill")?;
        let mut out = device.buffer_zeroed::<u32>(64)?;
        // SAFETY: each of the 64 threads writes its own element.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<64>::new(Dim3::x(1)),
                &mut out,
                |out| Ok([Argument::write(out)]),
            )
            .wait()?;
        }
        assert_eq!(out.read().wait(), (0..64).collect::<Vec<u32>>());
        // A device opened during the capture is not recorded and does not fail.
        Device::system_default()?;
        assert!(!trace.join("metadata").exists());
    }
    assert!(trace.join("metadata").is_file());
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn the_first_device_records_a_trace_until_it_drops() -> metal_oxide::Result<()> {
    if std::env::var_os(CHILD).is_some() {
        return record(Path::new(&std::env::var_os("METAL_OXIDE_CAPTURE").unwrap()));
    }
    let directory =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("capture-{}", std::process::id()));
    let trace = directory.join("fill.gputrace");

    let (success, text) = run(&trace, false);
    assert!(
        !success && text.contains("METAL_CAPTURE_ENABLED=1"),
        "{text}"
    );
    let (success, text) = run(&directory.join("fill.trace"), true);
    assert!(!success && text.contains("must end in .gputrace"), "{text}");

    let (success, text) = run(&trace, true);
    assert!(success, "{text}");
    let recorded = std::fs::read(trace.join("capture"))?;
    assert!(recorded.windows(13).any(|name| name == b"captured_fill"));

    // Metal does not replace an earlier trace.
    let (success, text) = run(&trace, true);
    assert!(!success && text.contains("Capture("), "{text}");
    std::fs::remove_dir_all(&directory)?;
    Ok(())
}
