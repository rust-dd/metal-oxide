use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTL4CommandQueue, MTL4Compiler, MTL4CompilerDescriptor, MTLCommandQueue, MTLDevice,
    MTLGPUFamily,
};

use crate::{Error, Result};

pub(super) enum Backend {
    Classic(Retained<ProtocolObject<dyn MTLCommandQueue>>),
    Metal4 {
        queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
        compiler: Retained<ProtocolObject<dyn MTL4Compiler>>,
    },
}

impl Backend {
    pub(super) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let requested = match std::env::var("METAL_OXIDE_BACKEND") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => String::new(),
            Err(error) => return Err(Error::Command(error.to_string())),
        };
        let metal4 = match requested.as_str() {
            "" => Self::supports_metal4(device),
            "classic" => false,
            "metal4" => {
                if !Self::supports_metal4(device) {
                    return Err(Error::UnsupportedDevice(
                        "Metal 4 requires macOS 26 and a Metal 4 GPU".into(),
                    ));
                }
                true
            }
            _ => {
                return Err(Error::Command(
                    "METAL_OXIDE_BACKEND must be classic or metal4".into(),
                ));
            }
        };
        if metal4 {
            let queue = device
                .newMTL4CommandQueue()
                .ok_or_else(|| Error::Command("could not create Metal 4 command queue".into()))?;
            let compiler = device
                .newCompilerWithDescriptor_error(&MTL4CompilerDescriptor::new())
                .map_err(|e| Error::Pipeline(e.localizedDescription().to_string()))?;
            Ok(Self::Metal4 { queue, compiler })
        } else {
            device
                .newCommandQueue()
                .map(Self::Classic)
                .ok_or_else(|| Error::Command("could not create command queue".into()))
        }
    }

    pub(super) fn supports_metal4(device: &ProtocolObject<dyn MTLDevice>) -> bool {
        objc2::available!(macos = 26.0) && device.supportsFamily(MTLGPUFamily::Metal4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Metal device on Apple Silicon"]
    fn requested_backend_uses_its_native_command_queue() {
        let device = crate::Device::system_default().unwrap();
        match std::env::var("METAL_OXIDE_BACKEND").as_deref() {
            Ok("metal4") => assert!(matches!(device.backend, Backend::Metal4 { .. })),
            Ok("classic") => assert!(matches!(device.backend, Backend::Classic(_))),
            _ => assert_eq!(
                matches!(device.backend, Backend::Metal4 { .. }),
                Backend::supports_metal4(&device.raw)
            ),
        }
    }
}
