use crate::{ABI_VERSION, Error, sha256};
use serde::{Deserialize, Serialize};

pub const COMPILER_NIGHTLY: &str = "nightly-2026-10-04";

/// Identity exchanged by the installed CLI and its compiler executable.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompilerInfo {
    pub protocol: u32,
    pub abi: u32,
    pub version: String,
    pub rustc: String,
    pub binary: String,
}

impl CompilerInfo {
    pub fn new(rustc: &str, binary: &[u8]) -> Self {
        Self {
            protocol: 1,
            abi: ABI_VERSION,
            version: env!("CARGO_PKG_VERSION").into(),
            rustc: rustc.trim().into(),
            binary: sha256(binary),
        }
    }

    pub fn from_json(json: &str) -> Result<Self, Error> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn to_json(&self) -> Result<String, Error> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn verify(&self, rustc: &str, binary: &[u8]) -> Result<(), Error> {
        let expected = Self::new(rustc, binary);
        for (field, actual, required) in [
            (
                "protocol",
                self.protocol.to_string(),
                expected.protocol.to_string(),
            ),
            ("abi", self.abi.to_string(), expected.abi.to_string()),
            ("version", self.version.clone(), expected.version),
            ("rustc", self.rustc.clone(), expected.rustc),
            ("binary", self.binary.clone(), expected.binary),
        ] {
            if actual != required {
                return Err(Error(format!(
                    "compiler {field} mismatch: expected {required:?}, got {actual:?}"
                )));
            }
        }
        Ok(())
    }
}
