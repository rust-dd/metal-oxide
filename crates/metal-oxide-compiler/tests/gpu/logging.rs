use std::process::{Child, Command, Output, Stdio};

use super::*;
use metal_oxide::Error;
use objc2_metal::{MTLDevice, MTLGPUFamily};

const CHILD: &str = "METAL_OXIDE_LOG_CHILD";

/// Whether the device logs; the Mac2-only paravirtual device of hosted macOS VMs cannot.
fn logs(device: &Device) -> bool {
    device.as_native().supportsFamily(MTLGPUFamily::Apple7)
}

/// In the child process, launches the logging fixture and returns true.
fn run_logging_kernel() -> bool {
    if std::env::var_os(CHILD).is_none() {
        return false;
    }
    let device = Device::system_default().unwrap();
    let (output, directory) =
        support::emit("crates/metal-oxide-compiler/tests/fixtures/logging.rs", &[]);
    support::checked(output);
    let module = support::artifact::module(&device, &directory);
    if !logs(&device) {
        let Err(Error::UnsupportedDevice(reason)) = &module else {
            panic!("expected no shader logging, got {:?}", module.err());
        };
        assert!(reason.contains("shader logging"), "{reason}");
        return true;
    }
    let pipeline = Pipeline::new(&device, &module.unwrap(), "logging").unwrap();
    let mut out = device.buffer_zeroed::<u32>(3).unwrap();
    // SAFETY: three threads write their own elements.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<3>::new(Dim3::x(1)),
            (&mut out, 0.1_f32, metal_oxide::F16::from_f32(0.5)),
            |(out, scale, half)| {
                Ok([
                    Argument::write(out),
                    Argument::value(*scale)?,
                    Argument::value(*half)?,
                ])
            },
        )
        .wait()
        .unwrap();
    }
    // Metal delivers shader logs asynchronously around completion.
    std::thread::sleep(std::time::Duration::from_millis(500));
    true
}

/// Runs `test` again in a child process that launches the logging fixture.
fn spawn_child(test: &str) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", test, "--nocapture"])
        .env(CHILD, "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

/// Checks that the child succeeded and ran its one test, since a filter matching none passes.
fn assert_ran(output: &Output) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}{stderr}");
    assert!(stdout.contains("test result: ok. 1 passed"), "{stdout}");
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn device_log_messages_reach_standard_error() {
    if run_logging_kernel() {
        return;
    }
    let output = spawn_child("logging::device_log_messages_reach_standard_error")
        .wait_with_output()
        .unwrap();
    assert_ran(&output);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let device = Device::system_default().unwrap();
    for thread in (0..3).filter(|_| logs(&device)) {
        let message = format!(
            "thread {thread} scale 0.100000001 half 0.5 wide 1099511627776 signed -3 -1099511627776 {{braces}} 100%"
        );
        assert!(stderr.contains(&message), "{stderr}");
    }
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn device_logs_survive_a_closed_standard_error() {
    if run_logging_kernel() {
        return;
    }
    let mut child = spawn_child("logging::device_logs_survive_a_closed_standard_error");
    // Closing the read end makes the child's writes to stderr fail with a broken pipe.
    drop(child.stderr.take());
    assert_ran(&child.wait_with_output().unwrap());
}
