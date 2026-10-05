use std::{marker::PhantomData, rc::Rc};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

use crate::{Error, Result};

use super::{Device, Module};

pub struct Pipeline {
    pub(super) raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) device_id: u64,
    pub(super) abi: Option<metal_oxide_artifact::Kernel>,
    marker: PhantomData<Rc<()>>,
}

impl Pipeline {
    pub fn new(device: &Device, module: &Module, entrypoint: &str) -> Result<Self> {
        if module.device_id != device.id() {
            return Err(Error::DeviceMismatch);
        }
        let abi = module
            .abi
            .as_ref()
            .map(|abi| {
                abi.kernels
                    .iter()
                    .find(|kernel| kernel.name == entrypoint)
                    .cloned()
                    .ok_or_else(|| Error::KernelNotFound(entrypoint.into()))
            })
            .transpose()?;
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
            abi,
            marker: PhantomData,
        })
    }

    pub fn thread_execution_width(&self) -> usize {
        self.raw.threadExecutionWidth()
    }

    pub fn max_threads_per_block(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }

    pub(super) fn validate_arguments(
        &self,
        block: crate::Dim3,
        arguments: &[super::Argument<'_>],
    ) -> Result<()> {
        if let Some(kernel) = &self.abi {
            if arguments.len() != kernel.parameters.len() {
                return Err(metal_oxide_artifact::Error(format!(
                    "kernel {} requires {} arguments, got {}",
                    kernel.name,
                    kernel.parameters.len(),
                    arguments.len()
                ))
                .into());
            }
            for (argument, parameter) in arguments.iter().zip(&kernel.parameters) {
                if argument.ty() != parameter.ty {
                    return Err(metal_oxide_artifact::Error(format!(
                        "kernel {} argument {} must be {:?}, got {:?}",
                        kernel.name,
                        parameter.name,
                        parameter.ty,
                        argument.ty()
                    ))
                    .into());
                }
            }
            if kernel
                .required_block
                .is_some_and(|expected| expected != [block.x, block.y, block.z])
            {
                return Err(crate::Error::InvalidLaunch(
                    "block does not match the kernel specialization",
                ));
            }
        }
        Ok(())
    }
}
