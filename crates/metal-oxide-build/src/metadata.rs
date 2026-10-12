use crate::process::{self, Result};
use serde::Deserialize;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

#[cfg(test)]
mod tests;

#[derive(Deserialize)]
pub(crate) struct Metadata {
    pub(crate) workspace_root: PathBuf,
    pub(crate) target_directory: PathBuf,
    pub(crate) packages: Vec<Package>,
}

#[derive(Deserialize)]
pub(crate) struct Package {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) version: String,
    pub(crate) manifest_path: PathBuf,
    pub(crate) targets: Vec<Target>,
}

#[derive(Deserialize)]
pub(crate) struct Target {
    pub(crate) name: String,
    pub(crate) kind: Vec<String>,
}

#[derive(Deserialize)]
struct UnitGraph {
    units: Vec<Unit>,
}

#[derive(Deserialize)]
struct Unit {
    pkg_id: String,
    target: Target,
    #[serde(default)]
    is_std: bool,
}

/// The kernel package and the Cargo workspace that resolves it.
pub(crate) struct Project {
    pub(crate) metadata: Metadata,
    kernel: usize,
}

/// Reads the workspace of the kernel package whose Cargo.toml is `manifest`.
pub(crate) fn load(manifest: &Path) -> Result<Project> {
    let manifest = manifest
        .canonicalize()
        .map_err(|error| format!("kernel manifest {}: {error}", manifest.display()))?;
    let json = process::capture(
        Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
            .args(["metadata", "--format-version", "1", "--locked"])
            .arg("--manifest-path")
            .arg(&manifest),
    )?;
    let metadata: Metadata = serde_json::from_str(&json)?;
    let kernel = metadata
        .packages
        .iter()
        .position(|p| p.manifest_path == manifest)
        .ok_or("kernel package must belong to its Cargo workspace")?;
    let targets = metadata.packages[kernel]
        .targets
        .iter()
        .filter(|t| t.kind.iter().any(|k| k == "lib" || k == "rlib"))
        .count();
    if targets != 1 {
        return Err("kernel package requires one library target".into());
    }
    Ok(Project { metadata, kernel })
}

impl Metadata {
    pub(crate) fn validate_units(&self, json: &str) -> Result<()> {
        let graph: UnitGraph = serde_json::from_str(json)?;
        for unit in graph.units.into_iter().filter(|unit| !unit.is_std) {
            let package = self
                .packages
                .iter()
                .find(|package| package.id == unit.pkg_id)
                .ok_or_else(|| format!("missing Cargo package for {}", unit.pkg_id))?;
            if unit.target.kind.iter().any(|kind| kind == "custom-build") {
                return Err(
                    format!("kernel build scripts are unsupported: {}", package.name).into(),
                );
            }
            if package.name != "metal-oxide-macros"
                && unit.target.kind.iter().any(|kind| kind == "proc-macro")
            {
                return Err(format!(
                    "kernel proc macros other than #[kernel] are unsupported: {}",
                    package.name
                )
                .into());
            }
        }
        Ok(())
    }
}

impl Project {
    pub(crate) fn kernel(&self) -> &Package {
        &self.metadata.packages[self.kernel]
    }

    pub(crate) fn kernel_crate(&self) -> &str {
        &self
            .kernel()
            .targets
            .iter()
            .find(|t| t.kind.iter().any(|k| k == "lib" || k == "rlib"))
            .unwrap()
            .name
    }

    /// The `metal-oxide-compiler` package beside the kernels' `metal-oxide-device`, as in a
    /// git checkout or a local path dependency of the metal-oxide repository.
    pub(crate) fn compiler_source(&self) -> Option<PathBuf> {
        self.metadata
            .packages
            .iter()
            .filter(|package| package.name == "metal-oxide-device")
            .filter_map(|package| {
                let source = package
                    .manifest_path
                    .parent()?
                    .parent()?
                    .join("metal-oxide-compiler/Cargo.toml");
                source.is_file().then_some(source)
            })
            .next()
    }
}
