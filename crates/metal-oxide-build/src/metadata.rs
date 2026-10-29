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
    pub(crate) edition: String,
    pub(crate) manifest_path: PathBuf,
    pub(crate) targets: Vec<Target>,
    dependencies: Vec<Dependency>,
}

#[derive(Deserialize)]
struct Dependency {
    name: String,
    rename: Option<String>,
    kind: Option<String>,
    uses_default_features: bool,
    features: Vec<String>,
}

/// How a package declares a dependency: the package it names and the features it enables.
pub(crate) struct Declaration {
    pub(crate) package: String,
    pub(crate) default_features: bool,
    pub(crate) features: Vec<String>,
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
    let metadata = workspace(&manifest, true)?;
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

/// The Cargo that runs this build, or the one on `PATH`.
pub(crate) fn cargo() -> Command {
    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

/// The Cargo workspace of the package whose Cargo.toml is `manifest`, with Cargo's
/// configuration from the package's directory; `dependencies` resolves the packages beyond
/// the workspace members.
pub(crate) fn workspace(manifest: &Path, dependencies: bool) -> Result<Metadata> {
    let mut command = cargo();
    command
        .current_dir(manifest.parent().unwrap())
        .args(["metadata", "--format-version", "1", "--locked"])
        .arg("--manifest-path")
        .arg(manifest);
    if !dependencies {
        command.arg("--no-deps");
    }
    Ok(serde_json::from_str(&process::capture(&mut command)?)?)
}

impl Metadata {
    /// The `metal-oxide-compiler` package beside `metal-oxide-device`, as in a git checkout or
    /// a local path dependency of the metal-oxide repository.
    pub(crate) fn compiler_source(&self) -> Option<PathBuf> {
        self.packages
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

    pub(crate) fn validate_units(&self, json: &str) -> Result<()> {
        let graph = serde_json::from_str::<UnitGraph>(json)?;
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

impl Package {
    /// The dependency that code names `name`, unified as Cargo unifies its normal declarations,
    /// or else its dev-dependency declarations, which only tests use.
    pub(crate) fn declaration(&self, name: &str) -> Result<Declaration> {
        let declared = |kind: Option<&str>| {
            self.dependencies
                .iter()
                .filter(|dependency| {
                    let code = dependency.rename.as_ref().unwrap_or(&dependency.name);
                    dependency.kind.as_deref() == kind && code.replace('-', "_") == name
                })
                .collect::<Vec<_>>()
        };
        let mut declarations = declared(None);
        if declarations.is_empty() {
            declarations = declared(Some("dev"));
        }
        let first = declarations.first().ok_or_else(|| {
            format!(
                "the kernels use {name}, which {} does not declare as a dependency",
                self.name
            )
        })?;
        let mut features = declarations
            .iter()
            .flat_map(|dependency| dependency.features.iter().cloned())
            .collect::<Vec<_>>();
        features.sort();
        features.dedup();
        Ok(Declaration {
            package: first.name.clone(),
            default_features: declarations
                .iter()
                .any(|dependency| dependency.uses_default_features),
            features,
        })
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
}
