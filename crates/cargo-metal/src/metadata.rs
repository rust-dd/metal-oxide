use crate::cli::Options;
use serde::Deserialize;
use std::{error::Error, path::PathBuf, process::Command};

#[derive(Deserialize)]
struct Metadata {
    workspace_root: PathBuf,
    packages: Vec<Package>,
}

#[derive(Deserialize)]
struct Package {
    name: String,
    manifest_path: PathBuf,
    metadata: serde_json::Value,
}

/// The selected host package and the kernel crate its `[package.metadata.metal]` names.
pub(crate) struct Host {
    pub(crate) name: String,
    pub(crate) manifest: PathBuf,
    pub(crate) workspace_root: PathBuf,
    pub(crate) kernels: PathBuf,
}

pub(crate) fn host(options: &Options) -> Result<Host, Box<dyn Error + Send + Sync>> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps", "--locked"])
        .arg("--manifest-path")
        .arg(&options.manifest)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    let metadata: Metadata = serde_json::from_slice(&output.stdout)?;
    let candidates = metadata
        .packages
        .into_iter()
        .filter(|p| {
            p.metadata.get("metal").is_some()
                && options.package.as_ref().is_none_or(|name| *name == p.name)
        })
        .collect::<Vec<_>>();
    let [package] = <[Package; 1]>::try_from(candidates)
        .map_err(|_| "select one host package with [package.metadata.metal] using -p HOST")?;
    let kernels = package.metadata["metal"]["kernels"]
        .as_str()
        .ok_or("package.metadata.metal.kernels must name a kernel Cargo.toml")?;
    Ok(Host {
        kernels: package.manifest_path.parent().unwrap().join(kernels),
        name: package.name,
        manifest: package.manifest_path,
        workspace_root: metadata.workspace_root,
    })
}
