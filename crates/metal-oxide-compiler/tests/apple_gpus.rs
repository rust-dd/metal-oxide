#![cfg(all(feature = "rustc-private", target_os = "macos"))]

mod support;

use metal_oxide_artifact::{Abi, ArtifactFile};
use std::path::{Path, PathBuf};

/// The kernels of each fixture compile for every Apple GPU that runs them. The tests of fixtures
/// that the compiler rejects, or that need configuration options, cover those separately.
#[test]
fn fixture_kernels_compile_for_each_apple_gpu() {
    let fixtures = support::root().join("crates/metal-oxide-compiler/tests/fixtures");
    let sources = [fixtures.clone(), fixtures.join("generated")]
        .iter()
        .flat_map(|directory| sources(directory))
        .collect::<Vec<_>>();
    let sdk = sdk_major();
    let results = parallel(&sources, |source| {
        compile(source, sdk).map_err(|error| format!("{}:\n{error}", source.display()))
    });
    let failures = results
        .iter()
        .filter_map(|result| result.as_ref().err().map(String::as_str))
        .collect::<Vec<_>>();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    let targets = results.into_iter().flatten().flatten().collect::<Vec<_>>();
    for generation in ["applegpu_g13", "applegpu_g14", "applegpu_g15"] {
        assert!(
            targets
                .iter()
                .any(|target| target.starts_with(generation) && target.ends_with("macOS 15")),
            "no {generation} target on macOS 15"
        );
    }
}

/// The targets that the kernels of `source` compiled for; none when the compiler rejects the
/// source or the SDK predates the macOS that its kernels require.
fn compile(source: &Path, sdk: u32) -> Result<Vec<String>, String> {
    let (output, directory) = support::emit(
        source.to_str().unwrap(),
        &["-C", "overflow-checks=off", "--metal-math-mode", "precise"],
    );
    if !output.status.success() {
        return Ok(Vec::new());
    }
    let abi = std::fs::read_to_string(directory.join(ArtifactFile::Abi.name())).unwrap();
    let abi = Abi::from_json(&abi).unwrap();
    if abi.minimum_macos() > sdk {
        return Ok(Vec::new());
    }
    metal_oxide_build::compile_msl(&directory).map_err(|error| error.to_string())?;
    let metallib = directory.join(ArtifactFile::Metallib.name());
    metal_oxide_build::compile_for_apple_gpus(&abi, &metallib).map_err(|error| error.to_string())
}

fn sources(directory: &Path) -> Vec<PathBuf> {
    let mut sources = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect::<Vec<_>>();
    sources.sort();
    sources
}

/// The major version of the macOS SDK that kernels build with.
fn sdk_major() -> u32 {
    let output = std::process::Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-version"])
        .output()
        .unwrap();
    let version = support::checked(output);
    version
        .trim()
        .split('.')
        .next()
        .unwrap()
        .parse::<u32>()
        .unwrap()
}

/// `check` of each item, spread over the machine's cores.
fn parallel<T: Sync, R: Send>(items: &[T], check: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    std::thread::scope(|scope| {
        let workers = items
            .chunks(items.len().div_ceil(threads).max(1))
            .map(|chunk| scope.spawn(|| chunk.iter().map(&check).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    })
}
