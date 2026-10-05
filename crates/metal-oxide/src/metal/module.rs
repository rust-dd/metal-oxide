use std::{marker::PhantomData, path::Path, rc::Rc};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLCompileOptions, MTLDevice, MTLLibrary, MTLMathFloatingPointFunctions, MTLMathMode,
};

use crate::{Error, Result};

use super::Device;

/// A Metal library loaded independently of Rust compiler internals.
pub struct Module {
    pub(super) raw: Retained<ProtocolObject<dyn MTLLibrary>>,
    pub(super) device_id: u64,
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
            marker: PhantomData,
        })
    }
}
