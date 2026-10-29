#![cfg(target_os = "macos")]

use metal_oxide_artifact::{ArtifactFile, Manifest};
use std::path::{Path, PathBuf};

/// Each example's saved kernels compile for every Apple GPU that runs them, so the libraries that
/// fetched and locked builds load work on GPUs that no test machine has.
#[test]
fn saved_example_kernels_compile_for_each_apple_gpu() {
    let mut artifacts = Vec::new();
    saved(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples"),
        &mut artifacts,
    );
    assert!(!artifacts.is_empty());
    let results = parallel(&artifacts, |directory| {
        let manifest = std::fs::read_to_string(directory.join(ArtifactFile::Manifest.name()));
        let manifest = Manifest::from_json(&manifest.unwrap()).unwrap();
        let metallib = directory.join(ArtifactFile::Metallib.name());
        metal_oxide_build::compile_for_apple_gpus(&manifest.abi, &metallib)
            .map_err(|error| format!("{}:\n{error}", directory.display()))
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
            "no {generation} target on macOS 15: {targets:?}"
        );
    }
}

/// Adds the saved artifact directories below `directory` to `found`.
fn saved(directory: &Path, found: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.join(ArtifactFile::Manifest.name()).is_file() {
            found.push(path);
        } else if path.is_dir() {
            saved(&path, found);
        }
    }
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
