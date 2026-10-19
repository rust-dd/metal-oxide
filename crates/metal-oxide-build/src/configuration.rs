use crate::{
    process::Result,
    recorded::{normalize, relative},
};
use metal_oxide_artifact::sha256;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Configuration {
    root: String,
    copied: bool,
    files: BTreeMap<String, String>,
    home: Option<String>,
    pub(crate) environment: BTreeMap<String, bool>,
}

impl Configuration {
    pub(crate) fn record(root: &Path, base: &Path, copied: bool) -> Result<Self> {
        let root = normalize(root);
        let home = cargo_home(&root);
        let mut files = BTreeMap::new();
        let mut environment = BTreeMap::new();
        for directory in ancestors(&root, copied, home.as_deref()) {
            if let Some(path) = existing(&directory)?
                && let Some(hash) = fingerprint(&path, &mut environment)?
            {
                files.insert(relative(&path, base).to_string_lossy().into_owned(), hash);
            }
        }
        let home = home
            .map(|home| existing(&home))
            .transpose()?
            .flatten()
            .map(|path| fingerprint(&path, &mut environment))
            .transpose()?
            .flatten();
        Ok(Self {
            root: relative(&root, base).to_string_lossy().into_owned(),
            copied,
            files,
            home,
            environment: environment
                .into_iter()
                .map(|(name, force)| (name, force.unwrap_or(false)))
                .collect(),
        })
    }

    pub(crate) fn current(&self, base: &Path) -> bool {
        Self::record(&base.join(&self.root), base, self.copied)
            .is_ok_and(|current| current == *self)
    }

    pub(crate) fn paths<'a>(&'a self, base: &'a Path) -> impl Iterator<Item = PathBuf> + 'a {
        let root = normalize(&base.join(&self.root));
        let home = cargo_home(&root);
        let paths = ancestors(&root, self.copied, home.as_deref())
            .chain(home.clone())
            .filter_map(|directory| existing(&directory).ok().flatten())
            .collect::<Vec<_>>();
        paths.into_iter()
    }
}

fn cargo_home(root: &Path) -> Option<PathBuf> {
    std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")))
        .map(|home| normalize(&root.join(home)))
}

fn ancestors<'a>(
    root: &'a Path,
    copied: bool,
    home: Option<&'a Path>,
) -> impl Iterator<Item = PathBuf> + 'a {
    // Copied workspaces generate only target-dir here; that file is not a portable input.
    root.ancestors()
        .skip(usize::from(copied))
        .map(|directory| directory.join(".cargo"))
        .filter(move |directory| Some(directory.as_path()) != home)
}

fn existing(directory: &Path) -> Result<Option<PathBuf>> {
    for name in ["config", "config.toml"] {
        let path = directory.join(name);
        if path.try_exists()? {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn fingerprint(
    path: &Path,
    configured: &mut BTreeMap<String, Option<bool>>,
) -> Result<Option<String>> {
    let text = std::fs::read_to_string(path)?;
    let mut settings = text
        .parse::<toml::Table>()
        .map_err(|_| format!("cannot parse Cargo configuration {}", path.display()))?;
    if settings.contains_key("include") {
        return Err(format!(
            "Cargo configuration includes are unsupported for kernel artifacts: {}; put the kernel settings in the discovered configuration file",
            path.display()
        ).into());
    }
    if let Some(environment) = settings.get("env").and_then(toml::Value::as_table) {
        for (name, value) in environment {
            let force = value.get("force").and_then(toml::Value::as_bool);
            configured
                .entry(name.clone())
                .and_modify(|previous| *previous = previous.or(force))
                .or_insert(force);
        }
    }
    for key in [
        "alias",
        "cache",
        "cargo-new",
        "credential-alias",
        "doc",
        "future-incompat-report",
        "http",
        "install",
        "net",
        "registries",
        "registry",
        "term",
    ] {
        settings.remove(key);
    }
    if let Some(build) = settings
        .get_mut("build")
        .and_then(toml::Value::as_table_mut)
    {
        for key in [
            "build-dir",
            "dep-info-basedir",
            "jobs",
            "rustdoc",
            "rustdocflags",
            "target-dir",
        ] {
            build.remove(key);
        }
    }
    prune_empty_tables(&mut settings);
    Ok((!settings.is_empty())
        .then(|| serde_json::to_vec(&settings))
        .transpose()?
        .map(|bytes| sha256(&bytes)))
}

fn prune_empty_tables(settings: &mut toml::Table) {
    settings.retain(|_, value| match value {
        toml::Value::Table(table) => {
            prune_empty_tables(table);
            !table.is_empty()
        }
        _ => true,
    });
}
