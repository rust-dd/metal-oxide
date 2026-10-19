use crate::process::Result;
use metal_oxide_artifact::sha256;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct Environment(BTreeMap<String, Option<String>>);

impl Environment {
    pub(crate) fn record(&mut self, info: &str, configured: &BTreeMap<String, bool>) -> Result<()> {
        for line in info
            .lines()
            .filter_map(|line| line.strip_prefix("# env-dep:"))
        {
            let (name, value) = match line.split_once('=') {
                Some((name, value)) => (unescape(name)?, Some(unescape(value)?)),
                None => (unescape(line)?, None),
            };
            if supplied_by_build(&name) || configured.get(&name) == Some(&true) {
                continue;
            }
            // Configuration fingerprints validate defaults; ambient values determine overrides.
            let value = if configured.contains_key(&name) {
                std::env::var(&name).ok()
            } else {
                value
            };
            let value = value.map(|value| sha256(value.as_bytes()));
            if let Some(previous) = self.0.insert(name.clone(), value.clone())
                && previous != value
            {
                return Err(format!(
                    "kernel packages read different values of environment variable {name}"
                )
                .into());
            }
        }
        Ok(())
    }

    pub(crate) fn current(&self) -> bool {
        self.0.iter().all(|(name, value)| {
            std::env::var(name)
                .ok()
                .map(|value| sha256(value.as_bytes()))
                == *value
        })
    }

    pub(crate) fn names(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }
}

fn supplied_by_build(name: &str) -> bool {
    crate::process::ISOLATED_ENV.contains(&name)
        || matches!(
            name,
            "CARGO"
                | "CARGO_BIN_NAME"
                | "CARGO_CRATE_NAME"
                | "CARGO_MANIFEST_DIR"
                | "CARGO_MANIFEST_PATH"
                | "CARGO_PRIMARY_PACKAGE"
                | "METAL_OXIDE_CARGO_TARGET"
                | "METAL_OXIDE_CARGO_KERNEL"
                | "METAL_OXIDE_CARGO_OUTPUT"
                | "METAL_OXIDE_CARGO_REMAP"
                | "METAL_OXIDE_CARGO_MATH_MODE"
                | "METAL_OXIDE_CARGO_SOURCE"
        )
        || name.strip_prefix("CARGO_PKG_").is_some_and(|key| {
            matches!(
                key,
                "AUTHORS"
                    | "DESCRIPTION"
                    | "HOMEPAGE"
                    | "LICENSE"
                    | "LICENSE_FILE"
                    | "NAME"
                    | "README"
                    | "REPOSITORY"
                    | "RUST_VERSION"
                    | "VERSION"
                    | "VERSION_MAJOR"
                    | "VERSION_MINOR"
                    | "VERSION_PATCH"
                    | "VERSION_PRE"
            )
        })
}

fn unescape(value: &str) -> Result<String> {
    let mut result = String::new();
    let mut characters = value.chars();
    while let Some(character) = characters.next() {
        result.push(if character == '\\' {
            match characters.next() {
                Some('n') => '\n',
                Some('r') => '\r',
                Some('\\') => '\\',
                _ => return Err("invalid escape in rustc's environment dependency".into()),
            }
        } else {
            character
        });
    }
    Ok(result)
}
