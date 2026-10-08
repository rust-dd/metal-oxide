use crate::{
    cli::Options,
    process::{self, Result},
};
use serde::Deserialize;
use std::{path::PathBuf, process::Command};

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
    pub(crate) manifest_path: PathBuf,
    pub(crate) targets: Vec<Target>,
    pub(crate) metadata: serde_json::Value,
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

pub(crate) struct Project {
    pub(crate) metadata: Metadata,
    pub(crate) host: usize,
    pub(crate) kernel: usize,
}

pub(crate) fn load(options: &Options) -> Result<Project> {
    let json = process::capture(
        Command::new("cargo")
            .args(["metadata", "--format-version", "1", "--locked"])
            .arg("--manifest-path")
            .arg(&options.manifest),
    )?;
    let metadata: Metadata = serde_json::from_str(&json)?;
    let candidates = metadata
        .packages
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            p.metadata.get("metal").is_some()
                && options.package.as_ref().is_none_or(|name| *name == p.name)
        })
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if candidates.len() != 1 {
        return Err("select one host package with [package.metadata.metal] using -p HOST".into());
    }
    let host = candidates[0];
    let package = &metadata.packages[host];
    let path = package.metadata["metal"]["kernels"]
        .as_str()
        .ok_or("package.metadata.metal.kernels must name a kernel Cargo.toml")?;
    let path = package
        .manifest_path
        .parent()
        .unwrap()
        .join(path)
        .canonicalize()?;
    let kernel = metadata
        .packages
        .iter()
        .position(|p| p.manifest_path == path)
        .ok_or("kernel package must belong to the host workspace")?;
    let targets = metadata.packages[kernel]
        .targets
        .iter()
        .filter(|t| t.kind.iter().any(|k| k == "lib" || k == "rlib"))
        .count();
    if targets != 1 {
        return Err("kernel package requires one library target".into());
    }
    Ok(Project {
        metadata,
        host,
        kernel,
    })
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
    pub(crate) fn host(&self) -> &Package {
        &self.metadata.packages[self.host]
    }
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
}
