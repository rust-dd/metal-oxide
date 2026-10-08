//! Versioned kernel metadata shared by the compiler, tools, and runtime.

mod compiler;
mod files;
mod layout;
mod model;
mod validate;

pub use compiler::{COMPILER_NIGHTLY, CompilerInfo};
pub use files::{ArtifactFile, COMPILER_OUTPUTS};
pub use layout::{FieldLayout, Layout, LayoutKind};
pub use model::*;

use sha2::{Digest, Sha256};
use std::fmt;

pub const ABI_VERSION: u32 = 3;
pub const DEVICE_TARGET: &str = "metal64-unknown-none";
pub const MSL_VERSION: &str = "3.1";

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for Error {}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

pub fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

impl Abi {
    pub fn from_json(json: &str) -> Result<Self, Error> {
        check_version(json, false)?;
        let abi: Self = serde_json::from_str(json)?;
        abi.validate()?;
        Ok(abi)
    }

    pub fn to_json(&self) -> Result<String, Error> {
        self.validate()?;
        Ok(serde_json::to_string_pretty(self)? + "\n")
    }
}

impl Manifest {
    pub fn from_json(json: &str) -> Result<Self, Error> {
        check_version(json, true)?;
        let manifest: Self = serde_json::from_str(json)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn to_json(&self) -> Result<String, Error> {
        self.validate()?;
        Ok(serde_json::to_string_pretty(self)? + "\n")
    }

    pub fn verify_library(&self, bytes: &[u8]) -> Result<(), Error> {
        self.validate()?;
        if sha256(bytes) != self.files.metallib {
            return Err(Error("metallib does not match the manifest hash".into()));
        }
        Ok(())
    }
}

fn check_version(json: &str, manifest: bool) -> Result<(), Error> {
    let value: serde_json::Value = serde_json::from_str(json)?;
    let value = if manifest { &value["abi"] } else { &value };
    if let Some(version) = value["version"].as_u64()
        && version != u64::from(ABI_VERSION)
    {
        return Err(Error(format!("unsupported kernel ABI version {version}")));
    }
    Ok(())
}
