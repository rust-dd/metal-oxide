#![allow(dead_code)]

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod artifact;
pub mod differential;
pub mod generate;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod gpu;
pub mod oracle;
pub mod process;

use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::OnceLock,
};

const COMPILER: &str = env!("CARGO_BIN_EXE_metal-oxide-compiler");

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// This process's output directory. The first call removes the directories of earlier runs,
/// keeping those changed within the hour, which a concurrent run may still use.
fn directory() -> PathBuf {
    static DIRECTORY: OnceLock<PathBuf> = OnceLock::new();
    DIRECTORY
        .get_or_init(|| {
            let parent = root().join("target/frontend-tests");
            let shared = dependencies();
            for entry in std::fs::read_dir(&parent).into_iter().flatten().flatten() {
                let stale = entry
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .ok()
                    .and_then(|modified| modified.elapsed().ok())
                    .is_some_and(|age| age > std::time::Duration::from_secs(3600));
                if stale && entry.path() != shared {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
            parent.join(std::process::id().to_string())
        })
        .clone()
}

/// The core, compiler builtins and device crates that kernels build against, shared by the test
/// processes of one nextest run; outside nextest each process prepares its own.
fn dependencies() -> PathBuf {
    let run = std::env::var("NEXTEST_RUN_ID").unwrap_or_else(|_| std::process::id().to_string());
    root()
        .join("target/frontend-tests")
        .join(format!("dependencies-{run}"))
}

fn compiler(source: &Path, name: &str, emit: &str) -> Command {
    compiler_in(source, name, emit, &directory())
}

fn compiler_in(source: &Path, name: &str, emit: &str, output: &Path) -> Command {
    std::fs::create_dir_all(output).unwrap();
    let extension = if emit == "metadata" { "rmeta" } else { "o" };
    let mut command = Command::new(COMPILER);
    command
        .arg(source)
        .args([
            "--crate-name",
            name,
            "--crate-type",
            "rlib",
            "--edition",
            "2024",
        ])
        .args(["--emit", emit])
        .arg("--target")
        .arg(root().join("crates/metal-oxide-build/targets/metal64-unknown-none.json"))
        .arg("-L")
        .arg(format!("dependency={}", dependencies().display()))
        .arg("-o")
        .arg(output.join(format!("lib{name}.{extension}")));
    command
}

fn dependency(command: &mut Command, name: &str) {
    command.arg("--extern").arg(format!(
        "{name}={}/lib{name}.rmeta",
        dependencies().display()
    ));
}

pub fn checked(output: Output) -> String {
    succeeded(output).unwrap_or_else(|failure| panic!("{failure}"))
}

/// The output's stdout, or its stdout and stderr when the process failed.
pub fn succeeded(output: Output) -> Result<String, String> {
    if output.status.success() {
        return Ok(String::from_utf8(output.stdout).unwrap());
    }
    Err(format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn prepare_dependencies() {
    static READY: OnceLock<()> = OnceLock::new();
    READY.get_or_init(|| {
        let shared = dependencies();
        std::fs::create_dir_all(&shared).unwrap();
        // The first of a run's concurrent processes prepares them; the others wait and reuse.
        let lock = std::fs::File::create(shared.join(".lock")).unwrap();
        lock.lock().unwrap();
        if shared.join(".ready").exists() {
            return;
        }
        let sysroot = Command::new("rustc")
            .args(["--print", "sysroot"])
            .output()
            .unwrap();
        let sysroot = PathBuf::from(checked(sysroot).trim());
        let library = sysroot.join("lib/rustlib/src/rust/library");
        // Kernel builds check core with `--release`, which disables its own overflow checks.
        let release = ["-Cdebug-assertions=off", "-Coverflow-checks=off"];
        let mut core = compiler_in(
            &library.join("core/src/lib.rs"),
            "core",
            "metadata",
            &shared,
        );
        checked(
            core.args(["--cap-lints", "allow"])
                .args(release)
                .output()
                .unwrap(),
        );

        let mut builtins = compiler_in(
            &library.join("compiler-builtins/compiler-builtins/src/lib.rs"),
            "compiler_builtins",
            "metadata",
            &shared,
        );
        dependency(&mut builtins, "core");
        checked(
            builtins
                .args([
                    "--cfg",
                    "feature=\"compiler-builtins\"",
                    "--cfg",
                    "feature=\"unmangled-names\"",
                    "--cap-lints",
                    "allow",
                ])
                .args(release)
                .output()
                .unwrap(),
        );

        let macro_library = shared.join(format!(
            "{}metal_oxide_macros{}",
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        ));
        checked(
            Command::new(sysroot.join("bin/rustc"))
                .arg(root().join("crates/metal-oxide-macros/src/lib.rs"))
                .args([
                    "--crate-name",
                    "metal_oxide_macros",
                    "--crate-type",
                    "proc-macro",
                    "--edition",
                    "2024",
                    "--extern",
                    "proc_macro",
                ])
                .arg("-o")
                .arg(&macro_library)
                .output()
                .unwrap(),
        );

        let mut device = compiler_in(
            &root().join("crates/metal-oxide-device/src/lib.rs"),
            "metal_oxide_device",
            "metadata",
            &shared,
        );
        dependency(&mut device, "core");
        dependency(&mut device, "compiler_builtins");
        checked(
            device
                .arg("--extern")
                .arg(format!("metal_oxide_macros={}", macro_library.display()))
                .output()
                .unwrap(),
        );
        std::fs::write(shared.join(".ready"), "").unwrap();
    });
}

pub fn compile(source: &str, emit: &str, device: bool) -> Output {
    if device {
        prepare_dependencies();
    }
    let path = root().join(source);
    let name = path.file_stem().unwrap().to_str().unwrap();
    let mut command = compiler(&path, name, emit);
    if device {
        for name in ["core", "compiler_builtins", "metal_oxide_device"] {
            dependency(&mut command, name);
        }
    }
    command.output().unwrap()
}

pub fn fixture(source: &str) -> Output {
    compile(
        &format!("crates/metal-oxide-compiler/tests/fixtures/{source}.rs"),
        "metadata",
        true,
    )
}

pub fn invalid_device(source: &str, client: &str, options: &[&str]) -> Output {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    prepare_dependencies();
    let alternate = directory().join(format!(
        "invalid-device-{}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let fixtures = root().join("crates/metal-oxide-compiler/tests/fixtures");
    let mut device = compiler_in(
        &fixtures.join(format!("{source}.rs")),
        "metal_oxide_device",
        "metadata",
        &alternate,
    );
    for name in ["core", "compiler_builtins"] {
        dependency(&mut device, name);
    }
    let macros = dependencies().join(format!(
        "{}metal_oxide_macros{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX,
    ));
    checked(
        device
            .arg("--extern")
            .arg(format!("metal_oxide_macros={}", macros.display()))
            .args(options)
            .output()
            .unwrap(),
    );
    let mut client = compiler(
        &fixtures.join(format!("{client}.rs")),
        "invalid_signature",
        "metadata",
    );
    for name in ["core", "compiler_builtins"] {
        dependency(&mut client, name);
    }
    client.arg("--extern").arg(format!(
        "metal_oxide_device={}/libmetal_oxide_device.rmeta",
        alternate.display()
    ));
    client
        .args(options)
        .arg("--metal-output")
        .arg(alternate.join("output"));
    client.output().unwrap()
}

pub fn rejected(output: Output, expected: &str) {
    assert!(
        !output.status.success(),
        "unexpected success: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains(expected), "expected {expected:?}:\n{error}");
    assert!(!error.contains("internal compiler error"), "{error}");
}

/// The id of the runtime check of `kind` at `line`, as the emitted MSL reports it.
pub fn check(directory: &Path, kind: serde_json::Value, line: u64) -> usize {
    let checks = serde_json::from_slice::<Vec<serde_json::Value>>(
        &std::fs::read(directory.join("checks.json")).unwrap(),
    )
    .unwrap();
    checks
        .iter()
        .position(|check| check["kind"] == kind && check["location"]["line"] == line)
        .unwrap_or_else(|| panic!("no {kind} check at line {line}: {checks:?}"))
}

pub fn emit(source: &str, options: &[&str]) -> (Output, PathBuf) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let output = directory().join(format!("output-{}", NEXT.fetch_add(1, Ordering::Relaxed)));
    (emit_into(source, options, &output), output)
}

pub fn emit_into(source: &str, options: &[&str], output: &Path) -> Output {
    prepare_dependencies();
    let path = root().join(source);
    let mut command = compiler(&path, "kernel", "metadata");
    for name in ["core", "compiler_builtins", "metal_oxide_device"] {
        dependency(&mut command, name);
    }
    command.arg("--metal-output").arg(output).args(options);
    process::output(&mut command, std::time::Duration::from_secs(60)).unwrap()
}

pub fn execute_msl(directory: &Path, main: &str) -> String {
    run_msl(directory, main).unwrap_or_else(|failure| panic!("{failure}"))
}

/// The output of `main` built for the host with the emitted MSL, or why the build or run failed.
pub fn run_msl(directory: &Path, main: &str) -> Result<String, String> {
    let source = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    let program = format!(
        "#include <iostream>\n{source}\nint main() {{ uint metal_oxide_gate_word = 0; atomic_uint metal_oxide_failure_word{{0}}; const device uint* metal_oxide_gate = &metal_oxide_gate_word; device atomic_uint* metal_oxide_failure = &metal_oxide_failure_word; {main} }}\n"
    );
    std::fs::write(directory.join("main.cpp"), program).unwrap();
    std::fs::write(
        directory.join("metal_stdlib"),
        include_str!("../../../metal-oxide-codegen/tests/support/metal_stdlib"),
    )
    .unwrap();
    std::fs::write(
        directory.join("metal_logging"),
        include_str!("../../../metal-oxide-codegen/tests/support/metal_logging"),
    )
    .unwrap();
    let mut compiler = Command::new("clang++");
    compiler
        .args(["-std=c++17", "-fsigned-char", "-Wno-unknown-attributes"])
        .arg("-I")
        .arg(directory)
        .arg(directory.join("main.cpp"))
        .arg("-o")
        .arg(directory.join("run"));
    let output = process::output(&mut compiler, std::time::Duration::from_secs(60))
        .map_err(|error| format!("clang++: {error}"))?;
    succeeded(output)?;
    let run = process::output(
        &mut Command::new(directory.join("run")),
        process::EXECUTION_TIMEOUT,
    )
    .map_err(|error| format!("the host build: {error}"))?;
    succeeded(run)
}
