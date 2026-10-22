use std::{fmt, path::Path, sync::Arc};

use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::{NSProcessInfo, NSString, NSURL};
use objc2_metal::{
    MTLCompileOptions, MTLDevice, MTLGPUFamily, MTLLibrary, MTLMathFloatingPointFunctions,
    MTLMathMode,
};

use crate::{Error, Result};

use super::Device;
use metal_oxide_artifact::{Abi, ArtifactFile, Check, Manifest};

/// A compiled Metal library and its optional kernel ABI.
pub struct Module {
    pub(super) raw: Retained<ProtocolObject<dyn MTLLibrary>>,
    pub(super) device_id: u64,
    pub(super) abi: Option<Abi>,
    /// The artifact's runtime checks, indexed by the ids its kernels record.
    pub(super) checks: Arc<[Check]>,
}

impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Module")
            .field("device_id", &self.device_id)
            .field("kernels", &self.abi.as_ref().map(|abi| abi.kernels.len()))
            .field("checks", &self.checks.len())
            .finish_non_exhaustive()
    }
}

impl Module {
    /// Compiles MSL with safe math and precise functions by default.
    ///
    /// Source pragmas and explicit math namespaces override these defaults.
    /// Generated MSL carries its selected numerical mode in the source.
    pub fn from_source(device: &Device, source: &str) -> Result<Self> {
        let raw = autoreleasepool(|_| {
            let options = MTLCompileOptions::new();
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            device
                .raw()
                .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
                .map_err(|error| Error::Library(error.localizedDescription().to_string()))
        })?;
        Ok(Self {
            raw,
            device_id: device.id(),
            abi: None,
            checks: Arc::new([]),
        })
    }

    pub fn from_metallib(device: &Device, path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().canonicalize()?;
        let path = path.to_str().ok_or(Error::InvalidPath)?;
        let raw = autoreleasepool(|_| {
            let url = NSURL::fileURLWithPath(&NSString::from_str(path));
            device
                .raw()
                .newLibraryWithURL_error(&url)
                .map_err(|error| Error::Library(error.localizedDescription().to_string()))
        })?;
        Ok(Self {
            raw,
            device_id: device.id(),
            abi: None,
            checks: Arc::new([]),
        })
    }

    /// Loads a precompiled library after checking its manifest and content hash.
    pub fn from_artifact(device: &Device, directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let manifest = Manifest::from_json(&std::fs::read_to_string(
            directory.join(ArtifactFile::Manifest.name()),
        )?)?;
        let bytes = std::fs::read(directory.join(ArtifactFile::Metallib.name()))?;
        Self::load_artifact(device, manifest, &bytes)
    }

    /// Loads embedded manifest JSON and metallib bytes after validating their ABI,
    /// capabilities and content hash. Neither input must outlive the module.
    pub fn from_bytes(device: &Device, manifest: &str, metallib: &[u8]) -> Result<Self> {
        Self::load_artifact(device, Manifest::from_json(manifest)?, metallib)
    }

    fn load_artifact(device: &Device, manifest: Manifest, bytes: &[u8]) -> Result<Self> {
        let macos = NSProcessInfo::processInfo()
            .operatingSystemVersion()
            .majorVersion;
        for feature in &manifest.abi.required_features {
            require(feature, macos, |family| device.raw().supportsFamily(family))?;
            if feature == "logging" {
                device.enable_logging()?;
            }
        }
        manifest.verify_library(bytes)?;
        let raw = autoreleasepool(|_| {
            let data = dispatch2::DispatchData::from_bytes(bytes);
            device
                .raw()
                .newLibraryWithData_error(&data)
                .map_err(|error| Error::Library(error.localizedDescription().to_string()))
        })?;
        Ok(Self {
            raw,
            device_id: device.id(),
            abi: Some(manifest.abi),
            checks: manifest.checks.into(),
        })
    }
    /// Checks that launch functions generated for the ABI with this digest match the artifact.
    pub fn verify_abi(&self, expected: &str) -> Result<()> {
        let abi = self.abi.as_ref().ok_or_else(|| {
            Error::Artifact(metal_oxide_artifact::Error(
                "module has no artifact ABI".into(),
            ))
        })?;
        if metal_oxide_artifact::sha256(abi.to_json()?.as_bytes()) != expected {
            return Err(Error::Artifact(metal_oxide_artifact::Error(
                "the launch functions do not match the artifact ABI".into(),
            )));
        }
        Ok(())
    }
}

/// Checks a validated artifact feature against the GPU families a device supports.
fn require(feature: &str, macos: isize, supports: impl Fn(MTLGPUFamily) -> bool) -> Result<()> {
    let requirement =
        metal_oxide_artifact::Capability::named(feature).expect("validated artifact capability");
    let family = MTLGPUFamily(MTLGPUFamily::Apple1.0 + requirement.apple_family as isize - 1);
    if macos >= requirement.macos as isize
        && (supports(family) || (requirement.mac2 && supports(MTLGPUFamily::Mac2)))
    {
        Ok(())
    } else {
        Err(Error::UnsupportedDevice(format!(
            "artifact requires {feature}"
        )))
    }
}

/// Matrix kernels assume full 32-lane SIMD groups, which every Apple7+ GPU has.
pub(super) fn require_simd_width(features: &[String], width: usize) -> Result<()> {
    if width != 32 && features.iter().any(|feature| feature == "simdgroup_matrix") {
        return Err(Error::UnsupportedDevice(
            "artifact requires 32-lane SIMD groups".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
