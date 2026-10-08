#![cfg(feature = "rustc-private")]

mod support;

use std::{io::ErrorKind, process::Command, time::Duration};

#[test]
fn timeout_terminates_descendants() {
    let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../target/timeout-test-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let started = directory.join("started");
    let survived = directory.join("survived");
    let mut command = Command::new("sh");
    command
        .args([
            "-c",
            "(printf started > \"$1\"; sleep 2; printf survived > \"$2\") & wait",
            "timeout-test",
        ])
        .arg(&started)
        .arg(&survived);
    let error = support::process::output(&mut command, Duration::from_secs(1)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::TimedOut);
    assert!(
        started.is_file(),
        "the descendant must start before the timeout"
    );
    std::thread::sleep(Duration::from_millis(1300));
    assert!(
        !survived.exists(),
        "the timed-out command left a descendant running"
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn completed_command_keeps_status_and_both_output_streams() {
    let mut command = Command::new("sh");
    command.args(["-c", "printf output; printf diagnostic >&2; exit 7"]);
    let output = support::process::output(&mut command, Duration::from_secs(5)).unwrap();
    assert_eq!(output.status.code(), Some(7));
    assert_eq!(output.stdout, b"output");
    assert_eq!(output.stderr, b"diagnostic");
}
