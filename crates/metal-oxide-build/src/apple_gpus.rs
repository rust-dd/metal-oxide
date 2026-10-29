use crate::{
    process::{self, Result},
    toolchain::Metal,
};
use metal_oxide_artifact::{Abi, Capability};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(test)]
mod tests;

impl Metal {
    /// Compiles `metallib` with the offline compiler for each Apple GPU that runs `abi`'s
    /// kernels, on the ABI's minimum macOS and on the SDK's, and returns the targets.
    pub(crate) fn compile_for_apple_gpus(&self, abi: &Abi, metallib: &Path) -> Result<Vec<String>> {
        // The offline compiler lacks the logging runtime that converts float arguments.
        if abi
            .required_features
            .iter()
            .any(|feature| feature == "logging")
        {
            return Ok(Vec::new());
        }
        let sdk = self
            .sdk
            .split('.')
            .next()
            .unwrap_or_default()
            .parse::<u32>()?;
        let minimum = abi.minimum_macos();
        if minimum > sdk {
            return Err(format!("the macOS {} SDK cannot target macOS {minimum}", self.sdk).into());
        }
        let family = abi
            .required_features
            .iter()
            .filter_map(|feature| Capability::named(feature))
            .map(|capability| capability.apple_family)
            .max()
            .unwrap_or(1);
        let scratch = Scratch::new()?;
        // The offline compiler reads JSON only from a script with this extension.
        let script = scratch.0.join("pipelines.mtlp-json");
        std::fs::write(&script, serde_json::to_vec(&pipelines(abi, metallib)?)?)?;
        let mut targets = Vec::new();
        let mut failures = Vec::new();
        for macos in BTreeSet::from([minimum, sdk]) {
            let version = format!("{macos}.0");
            let archs = process::capture(self.tool("applegpu-nt").args([
                "-available-archs",
                "macos",
                &version,
            ]))?;
            for arch in archs
                .split_whitespace()
                .filter(|arch| supports(arch, family))
            {
                let output = self
                    .tool("applegpu-nt")
                    .args([
                        "-arch",
                        arch,
                        "-platform_version",
                        "macos",
                        &version,
                        &self.sdk,
                    ])
                    .arg("-N")
                    .arg(&script)
                    .arg(metallib)
                    .arg("-o")
                    .arg(scratch.0.join("library.bin"))
                    .output()?;
                let target = format!("{arch} on macOS {macos}");
                if output.status.success() {
                    targets.push(target);
                } else {
                    let message = String::from_utf8_lossy(&output.stderr);
                    failures.push(format!("{target}: {}", message.trim()));
                }
            }
        }
        if failures.is_empty() {
            Ok(targets)
        } else {
            Err(failures.join("\n").into())
        }
    }
}

/// Whether `arch`, an offline compiler target such as `applegpu_g13g`, is an Apple GPU of at
/// least Apple `family`; G13 (M1) is Apple7, G14 (M2) Apple8 and the M4 Max Apple9.
fn supports(arch: &str, family: u32) -> bool {
    let generation = arch
        .strip_prefix("applegpu_g")
        .map(|name| name.trim_end_matches(|character: char| character.is_ascii_alphabetic()))
        .and_then(|generation| generation.parse::<u32>().ok());
    let apple = match generation {
        Some(13) => 7,
        Some(14) => 8,
        Some(15..) => 9,
        _ => return false,
    };
    apple >= family
}

/// A pipeline script that names each kernel of the library at `metallib`.
fn pipelines(abi: &Abi, metallib: &Path) -> Result<serde_json::Value> {
    let pipelines = abi
        .kernels
        .iter()
        .map(|kernel| {
            serde_json::json!({ "compute_function": format!("alias:library#{}", kernel.name) })
        })
        .collect::<Vec<_>>();
    let library = serde_json::json!({ "label": "library", "path": std::path::absolute(metallib)? });
    Ok(serde_json::json!({
        "libraries": { "paths": [library] },
        "pipelines": { "compute_pipelines": pipelines },
    }))
}

/// A directory for the offline compiler's script and output, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Result<Self> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "metal-oxide-apple-gpus-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory)?;
        Ok(Self(directory))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
