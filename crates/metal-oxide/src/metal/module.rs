use std::{marker::PhantomData, path::Path, rc::Rc};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLCompileOptions, MTLDevice, MTLGPUFamily, MTLLibrary, MTLMathFloatingPointFunctions,
    MTLMathMode,
};

use crate::{Error, Result};

use super::Device;
use metal_oxide_artifact::{Abi, ArtifactFile, Manifest};

/// A compiled Metal library and its optional kernel ABI.
pub struct Module {
    pub(super) raw: Retained<ProtocolObject<dyn MTLLibrary>>,
    pub(super) device_id: u64,
    pub(super) abi: Option<Abi>,
    marker: PhantomData<Rc<()>>,
}

impl Module {
    /// Compiles MSL with fast math disabled.
    pub fn from_source(device: &Device, source: &str) -> Result<Self> {
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        let raw = device
            .raw
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .map_err(|error| Error::Library(error.localizedDescription().to_string()))?;
        Ok(Self {
            raw,
            device_id: device.id(),
            abi: None,
            marker: PhantomData,
        })
    }

    pub fn from_metallib(device: &Device, path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().canonicalize()?;
        let path = path.to_str().ok_or(Error::InvalidPath)?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(path));
        let raw = device
            .raw
            .newLibraryWithURL_error(&url)
            .map_err(|error| Error::Library(error.localizedDescription().to_string()))?;
        Ok(Self {
            raw,
            device_id: device.id(),
            abi: None,
            marker: PhantomData,
        })
    }

    /// Loads a precompiled library after checking its manifest and content hash.
    pub fn from_artifact(device: &Device, directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let manifest = Manifest::from_json(&std::fs::read_to_string(
            directory.join(ArtifactFile::Manifest.name()),
        )?)?;
        for feature in &manifest.abi.required_features {
            let (family, reason) = match feature.as_str() {
                "simd_groups" => (MTLGPUFamily::Apple7, "artifact requires SIMD-group support"),
                "int32_atomics" => (
                    MTLGPUFamily::Apple1,
                    "artifact requires 32-bit integer atomics",
                ),
                _ => unreachable!("validated artifact capabilities"),
            };
            if !device.raw.supportsFamily(family) {
                return Err(Error::UnsupportedDevice(reason.into()));
            }
        }
        let bytes = std::fs::read(directory.join(ArtifactFile::Metallib.name()))?;
        manifest.verify_library(&bytes)?;
        let data = dispatch2::DispatchData::from_bytes(&bytes);
        let raw = device
            .raw
            .newLibraryWithData_error(&data)
            .map_err(|error| Error::Library(error.localizedDescription().to_string()))?;
        Ok(Self {
            raw,
            device_id: device.id(),
            abi: Some(manifest.abi),
            marker: PhantomData,
        })
    }
    /// Checks that generated bindings describe this artifact's exact ABI.
    pub fn verify_abi(&self, expected: &str) -> Result<()> {
        let abi = self.abi.as_ref().ok_or_else(|| {
            Error::Artifact(metal_oxide_artifact::Error(
                "module has no artifact ABI".into(),
            ))
        })?;
        if metal_oxide_artifact::sha256(abi.to_json()?.as_bytes()) != expected {
            return Err(Error::Artifact(metal_oxide_artifact::Error(
                "generated bindings do not match the artifact ABI".into(),
            )));
        }
        Ok(())
    }
}
