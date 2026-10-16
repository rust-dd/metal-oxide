use std::sync::Arc;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTL4Compiler, MTL4CompilerDescriptor, MTLDevice, MTLGPUFamily};

use super::{
    classic::ClassicQueue,
    metal4::BatchCache,
    queue::{Metal4Queue, QUEUES, Queue, Signals},
};
use crate::{Error, Result};

pub(super) enum Backend {
    Classic {
        queues: Vec<Queue<ClassicQueue>>,
        signals: Arc<Signals>,
    },
    Metal4 {
        queues: Vec<Queue<Metal4Queue>>,
        compiler: Retained<ProtocolObject<dyn MTL4Compiler>>,
        cache: BatchCache,
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
            let queues = (0..QUEUES)
                .map(|_| Metal4Queue::new(device).map(Queue::new))
                .collect::<Result<Vec<_>>>()?;
            let compiler = device
                .newCompilerWithDescriptor_error(&MTL4CompilerDescriptor::new())
                .map_err(|e| Error::Pipeline(e.localizedDescription().to_string()))?;
            Ok(Self::Metal4 {
                queues,
                compiler,
                cache: BatchCache::default(),
            })
        } else {
            let queues = (0..QUEUES)
                .map(|_| ClassicQueue::new(device).map(Queue::new))
                .collect::<Result<Vec<_>>>()?;
            Ok(Self::Classic {
                queues,
                signals: Arc::default(),
            })
        }
    }

    /// The number of batches in flight on each queue.
    pub(super) fn running(&self) -> [usize; QUEUES] {
        match self {
            Self::Classic { queues, .. } => std::array::from_fn(|queue| queues[queue].running()),
            Self::Metal4 { queues, .. } => std::array::from_fn(|queue| queues[queue].running()),
        }
    }

    pub(super) fn supports_metal4(device: &ProtocolObject<dyn MTLDevice>) -> bool {
        objc2::available!(macos = 26.0) && device.supportsFamily(MTLGPUFamily::Metal4)
    }
}

#[cfg(test)]
mod tests;
