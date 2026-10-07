use crate::{metadata::Project, process::Result};
use metal_oxide_artifact::sha256;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

#[cfg(test)]
mod tests;

#[derive(Debug, PartialEq, Eq, Serialize)]
pub(crate) struct Inputs {
    files: BTreeMap<PathBuf, String>,
    features: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Deserialize)]
#[serde(tag = "reason")]
enum Message {
    #[serde(rename = "compiler-artifact")]
    Artifact {
        package_id: String,
        filenames: Vec<PathBuf>,
        features: Vec<String>,
    },
    #[serde(other)]
    Other,
}

impl Inputs {
    pub(crate) fn from_cargo(output: &str, project: &Project) -> Result<Self> {
        let mut inputs = Self {
            files: BTreeMap::new(),
            features: BTreeMap::new(),
        };
        for line in output.lines() {
            if let Message::Artifact {
                package_id,
                filenames,
                features,
            } = serde_json::from_str(line)?
            {
                for file in filenames {
                    inputs.hash(&file)?;
                }
                inputs
                    .features
                    .entry(package_id)
                    .or_default()
                    .extend(features);
            }
        }
        if !inputs.features.contains_key(&project.kernel().id) {
            return Err("Cargo did not produce kernel metadata".into());
        }
        for package in &project.metadata.packages {
            if inputs.features.contains_key(&package.id) {
                inputs.hash(&package.manifest_path)?;
            }
        }
        for name in ["Cargo.toml", "Cargo.lock"] {
            inputs.hash(&project.metadata.workspace_root.join(name))?;
        }
        for ancestor in project.metadata.workspace_root.ancestors() {
            for name in [".cargo/config", ".cargo/config.toml"] {
                let path = ancestor.join(name);
                if path.is_file() {
                    inputs.hash(&path)?;
                }
            }
        }
        Ok(inputs)
    }

    fn hash(&mut self, path: &Path) -> Result<()> {
        let bytes = std::fs::read(path).map_err(|error| {
            format!("cannot read Cargo build input {}: {error}", path.display())
        })?;
        self.files.insert(path.to_owned(), sha256(&bytes));
        Ok(())
    }
}
