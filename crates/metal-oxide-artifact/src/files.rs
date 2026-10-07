use crate::{Files, sha256};
use std::{io, path::Path};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactFile {
    Msl,
    OxideIr,
    Abi,
    Bindings,
    Air,
    Metallib,
    Manifest,
}

impl ArtifactFile {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Msl => "kernels.metal",
            Self::OxideIr => "kernels.oxide-ir",
            Self::Abi => "abi.json",
            Self::Bindings => "bindings.rs",
            Self::Air => "kernels.ir",
            Self::Metallib => "kernels.metallib",
            Self::Manifest => "manifest.json",
        }
    }
}

pub const COMPILER_OUTPUTS: [ArtifactFile; 4] = [
    ArtifactFile::Msl,
    ArtifactFile::OxideIr,
    ArtifactFile::Abi,
    ArtifactFile::Bindings,
];

impl Files {
    pub fn from_directory(directory: impl AsRef<Path>) -> io::Result<Self> {
        let hash = |file: ArtifactFile| {
            std::fs::read(directory.as_ref().join(file.name())).map(|bytes| sha256(&bytes))
        };
        Ok(Self {
            msl: hash(ArtifactFile::Msl)?,
            oxide_ir: hash(ArtifactFile::OxideIr)?,
            ir: hash(ArtifactFile::Air)?,
            metallib: hash(ArtifactFile::Metallib)?,
            bindings: hash(ArtifactFile::Bindings)?,
        })
    }

    pub fn entries(&self) -> [(ArtifactFile, &str); 5] {
        [
            (ArtifactFile::Msl, &self.msl),
            (ArtifactFile::OxideIr, &self.oxide_ir),
            (ArtifactFile::Air, &self.ir),
            (ArtifactFile::Metallib, &self.metallib),
            (ArtifactFile::Bindings, &self.bindings),
        ]
    }
}
