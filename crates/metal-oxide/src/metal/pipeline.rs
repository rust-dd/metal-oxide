use std::sync::{Arc, Mutex, OnceLock};

use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4Compiler, MTL4ComputePipelineDescriptor, MTL4LibraryFunctionDescriptor,
    MTLComputePipelineState, MTLDevice, MTLLibrary,
};

use crate::{Error, Result};

use super::{Device, Module, argument::Signature, backend::Backend, failures::KernelChecks};

static METAL4_COMPILATION: Mutex<()> = Mutex::new(());

/// A compiled kernel of one device. Clones are cheap handles to the same native pipeline.
#[derive(Clone)]
pub struct Pipeline(Arc<Compiled>);

struct Compiled {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    device: Device,
    abi: Option<metal_oxide_artifact::Kernel>,
    /// Present for generated kernels, which bind gate and failure words.
    checks: Option<Arc<KernelChecks>>,
    /// The first argument signature validated for each parameter; others are checked per launch.
    validated: Box<[OnceLock<Signature>]>,
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
        let raw = compile(device, module, entrypoint)?;
        if raw.staticThreadgroupMemoryLength() > device.raw().maxThreadgroupMemoryLength() {
            return Err(Error::InvalidLaunch(
                "kernel exceeds threadgroup memory capacity",
            ));
        }
        Ok(Self(Arc::new(Compiled {
            raw,
            device: device.clone(),
            validated: (0..abi.as_ref().map_or(0, |abi| abi.parameters.len()))
                .map(|_| OnceLock::new())
                .collect(),
            checks: abi.as_ref().map(|kernel| {
                Arc::new(KernelChecks {
                    name: kernel.name.clone(),
                    gate_binding: kernel.gate_binding,
                    fallible: kernel.fallible,
                    lengths: kernel.lengths,
                    checks: Arc::clone(&module.checks),
                })
            }),
            abi,
        })))
    }

    pub fn thread_execution_width(&self) -> usize {
        self.0.raw.threadExecutionWidth()
    }

    pub fn max_threads_per_block(&self) -> usize {
        self.0.raw.maxTotalThreadsPerThreadgroup()
    }

    pub(super) fn raw(&self) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        &self.0.raw
    }

    pub(super) fn device(&self) -> &Device {
        &self.0.device
    }

    pub(super) fn checks(&self) -> Option<&Arc<KernelChecks>> {
        self.0.checks.as_ref()
    }

    pub(super) fn validate_launch(
        &self,
        config: crate::DynamicLaunchConfig,
        arguments: &[super::Argument<'_>],
    ) -> Result<()> {
        if let Some(kernel) = &self.0.abi {
            if arguments.len() != kernel.parameters.len() {
                return Err(metal_oxide_artifact::Error(format!(
                    "kernel {} requires {} arguments, got {}",
                    kernel.name,
                    kernel.parameters.len(),
                    arguments.len()
                ))
                .into());
            }
            for ((argument, parameter), checked) in arguments
                .iter()
                .zip(&kernel.parameters)
                .zip(&self.0.validated)
            {
                if checked.get() == Some(&argument.signature()) {
                    continue;
                }
                let ty = argument.ty()?;
                if ty != parameter.ty {
                    return Err(metal_oxide_artifact::Error(format!(
                        "kernel {} argument {} must be {:?}, got {:?}",
                        kernel.name, parameter.name, parameter.ty, ty
                    ))
                    .into());
                }
                let _ = checked.set(argument.signature());
            }
            let (grid, block) = (config.grid, config.block);
            if kernel
                .required_block
                .is_some_and(|expected| expected != [block.x, block.y, block.z])
            {
                return Err(crate::Error::InvalidLaunch(
                    "block does not match the kernel specialization",
                ));
            }
            if kernel.one_dimensional && [grid.y, grid.z, block.y, block.z] != [1; 4] {
                return Err(crate::Error::InvalidLaunch(
                    "a kernel that reads its thread index requires a one-dimensional launch",
                ));
            }
        }
        Ok(())
    }
}

/// Compiles a module function for the device's native command path.
pub(super) fn compile(
    device: &Device,
    module: &Module,
    entrypoint: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    autoreleasepool(|_| {
        // Concurrent Metal 4 pipeline creation has crashed inside the driver on
        // macOS 26.2, even with independent device contexts.
        let _compilation = matches!(device.backend(), Backend::Metal4 { .. }).then(|| {
            METAL4_COMPILATION
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
        });
        let function = module
            .raw
            .newFunctionWithName(&NSString::from_str(entrypoint))
            .ok_or_else(|| Error::KernelNotFound(entrypoint.into()))?;
        match device.backend() {
            Backend::Classic { .. } => device
                .raw()
                .newComputePipelineStateWithFunction_error(&function),
            Backend::Metal4 { compiler, .. } => {
                let descriptor = MTL4ComputePipelineDescriptor::new();
                let function = MTL4LibraryFunctionDescriptor::new();
                function.setName(Some(&NSString::from_str(entrypoint)));
                function.setLibrary(Some(&module.raw));
                descriptor.setComputeFunctionDescriptor(Some(&function));
                compiler.newComputePipelineStateWithDescriptor_compilerTaskOptions_error(
                    &descriptor,
                    None,
                )
            }
        }
        .map_err(|error| Error::Pipeline(error.localizedDescription().to_string()))
    })
}
