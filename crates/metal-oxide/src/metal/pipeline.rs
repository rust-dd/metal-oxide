use std::{marker::PhantomData, rc::Rc};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

use crate::{Error, Result};

use super::{Device, Module};

pub struct Pipeline {
    pub(super) raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) device_id: u64,
    marker: PhantomData<Rc<()>>,
}

impl Pipeline {
    pub fn new(device: &Device, module: &Module, entrypoint: &str) -> Result<Self> {
        if module.device_id != device.id() {
            return Err(Error::DeviceMismatch);
        }
        let function = module
            .raw
            .newFunctionWithName(&NSString::from_str(entrypoint))
            .ok_or_else(|| Error::KernelNotFound(entrypoint.into()))?;
        let raw = device
            .raw
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| Error::Pipeline(error.localizedDescription().to_string()))?;
        Ok(Self {
            raw,
            device_id: device.id(),
            marker: PhantomData,
        })
    }

    pub fn thread_execution_width(&self) -> usize {
        self.raw.threadExecutionWidth()
    }

    pub fn max_threads_per_threadgroup(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }
}
