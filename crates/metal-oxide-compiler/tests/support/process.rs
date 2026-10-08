use std::{
    fs::File,
    io,
    os::unix::process::CommandExt,
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

pub fn output(command: &mut Command, timeout: Duration) -> io::Result<Output> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory = super::root().join(format!(
        "target/process-tests/{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory)?;
    let stdout = directory.join("stdout");
    let stderr = directory.join("stderr");
    let mut child = command
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(File::create(&stdout)?)
        .stderr(File::create(&stderr)?)
        .spawn()?;
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() >= timeout {
            let _ = Command::new("/bin/kill")
                .args(["-KILL", "--", &format!("-{}", child.id())])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            let _ = child.kill();
            child.wait()?;
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "{command:?} exceeded {timeout:?}; logs: {}",
                    directory.display()
                ),
            ));
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let output = Output {
        status,
        stdout: std::fs::read(stdout)?,
        stderr: std::fs::read(stderr)?,
    };
    if status.success() {
        std::fs::remove_dir_all(directory)?;
    }
    Ok(output)
}
