use crate::{
    cli::Options,
    process::{self, Result},
};
use serde::Deserialize;
use std::{collections::HashSet, path::PathBuf, process::Command};

#[cfg(test)]
mod tests;

#[derive(Deserialize)]
pub(crate) struct Metadata {
    pub(crate) workspace_root: PathBuf,
    pub(crate) target_directory: PathBuf,
    pub(crate) packages: Vec<Package>,
    pub(crate) resolve: Resolve,
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
pub(crate) struct Resolve {
    pub(crate) nodes: Vec<Node>,
}

#[derive(Deserialize)]
pub(crate) struct Node {
    pub(crate) id: String,
    pub(crate) deps: Vec<Dependency>,
}

#[derive(Deserialize)]
pub(crate) struct Dependency {
    pub(crate) pkg: String,
    pub(crate) dep_kinds: Vec<DependencyKind>,
}

#[derive(Deserialize)]
pub(crate) struct DependencyKind {
    pub(crate) kind: Option<String>,
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
    let mut dependencies = HashSet::new();
    let mut pending = vec![metadata.packages[kernel].id.clone()];
    while let Some(id) = pending.pop() {
        if dependencies.insert(id.clone()) {
            let node = metadata
                .resolve
                .nodes
                .iter()
                .find(|n| n.id == id)
                .ok_or("missing Cargo resolve node")?;
            pending.extend(
                node.deps
                    .iter()
                    .filter(|dep| {
                        dep.dep_kinds
                            .iter()
                            .any(|kind| kind.kind.as_deref() != Some("dev"))
                    })
                    .map(|dep| dep.pkg.clone()),
            );
        }
    }
    for package in &metadata.packages {
        if dependencies.contains(&package.id) {
            if package
                .targets
                .iter()
                .any(|t| t.kind.iter().any(|k| k == "custom-build"))
            {
                return Err(
                    format!("kernel build scripts are unsupported: {}", package.name).into(),
                );
            }
            if package.name != "metal-oxide-macros"
                && package
                    .targets
                    .iter()
                    .any(|t| t.kind.iter().any(|k| k == "proc-macro"))
            {
                return Err(format!(
                    "kernel proc macros other than #[kernel] are unsupported: {}",
                    package.name
                )
                .into());
            }
        }
    }
    Ok(Project {
        metadata,
        host,
        kernel,
    })
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
