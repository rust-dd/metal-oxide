use std::{
    fmt,
    sync::{Arc, Mutex, OnceLock},
};

use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4Compiler, MTL4ComputePipelineDescriptor, MTL4LibraryFunctionDescriptor,
    MTLComputePipelineState, MTLDevice, MTLLibrary,
};

use crate::{Error, Result, launch::shape};

use super::{Device, Module, argument::Signature, backend::Backend, failures::KernelChecks};

mod source;
pub(super) use source::{Kernel, Source};

static METAL4_COMPILATION: Mutex<()> = Mutex::new(());

/// A compiled kernel of one device. Clones are cheap handles to the same native pipeline.
#[derive(Clone)]
pub struct Pipeline {
    device: Device,
    compiled: Arc<Compiled>,
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
            .field("entrypoint", &self.compiled.entrypoint)
            .field("device_id", &self.device.id())
            .field("thread_execution_width", &self.thread_execution_width())
            .field("max_threads_per_block", &self.max_threads_per_block())
            .finish_non_exhaustive()
    }
}

/// A native pipeline and its ABI, without the device, so a device can cache it.
pub(super) struct Compiled {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    entrypoint: Box<str>,
    abi: Option<metal_oxide_artifact::Kernel>,
    /// Present for generated kernels, which bind gate and failure words.
    checks: Option<Arc<KernelChecks>>,
    /// The first argument signature validated for each parameter; others are checked per launch.
    validated: Box<[OnceLock<Signature>]>,
}

impl Pipeline {
    /// Compiles the kernel `entrypoint` of `module` for the device's GPU.
    pub fn new(device: &Device, module: &Module, entrypoint: &str) -> Result<Self> {
        let compiled = Compiled::new(device, module, entrypoint)?;
        Ok(Self::from_compiled(device.clone(), Arc::new(compiled)))
    }

    pub(super) fn from_compiled(device: Device, compiled: Arc<Compiled>) -> Self {
        Self { device, compiled }
    }

    /// The SIMD-group width: how many threads execute together, 32 on Apple GPUs.
    pub fn thread_execution_width(&self) -> usize {
        self.compiled.raw.threadExecutionWidth()
    }

    /// The most threads in one block of this pipeline, which its register use can lower.
    pub fn max_threads_per_block(&self) -> usize {
        self.compiled.raw.maxTotalThreadsPerThreadgroup()
    }

    pub(super) fn raw(&self) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        &self.compiled.raw
    }

    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    pub(super) fn checks(&self) -> Option<&Arc<KernelChecks>> {
        self.compiled.checks.as_ref()
    }

    pub(super) fn validate_launch(
        &self,
        config: crate::DynamicLaunchConfig,
        arguments: &[super::argument::ArgumentInfo],
    ) -> Result<()> {
        if let Some(kernel) = &self.compiled.abi {
            if arguments.len() != kernel.parameters.len() {
                return Err(crate::Error::InvalidLaunch(format!(
                    "kernel {} takes {} arguments, got {}",
                    kernel.name,
                    kernel.parameters.len(),
                    arguments.len()
                )));
            }
            for ((argument, parameter), checked) in arguments
                .iter()
                .zip(&kernel.parameters)
                .zip(&self.compiled.validated)
            {
                if checked.get() == Some(&argument.signature()) {
                    continue;
                }
                let ty = argument.ty()?;
                if ty != parameter.ty {
                    return Err(crate::Error::InvalidLaunch(format!(
                        "argument {} of kernel {} takes a {}, got a {ty}",
                        parameter.name, kernel.name, parameter.ty
                    )));
                }
                let _ = checked.set(argument.signature());
            }
            for contract in &kernel.requires {
                let length = |index: usize| arguments[index].length().unwrap_or_default() as u64;
                let value = |index: usize| {
                    let scalar = match &kernel.parameters[index].ty {
                        metal_oxide_artifact::ParameterType::Value { layout } => {
                            layout.scalar_type()
                        }
                        _ => None,
                    };
                    scalar
                        .and_then(|scalar| arguments[index].integer(scalar))
                        .unwrap_or_default()
                };
                if !contract.holds(length, value) {
                    return Err(crate::Error::Contract {
                        kernel: kernel.name.clone(),
                        relation: contract.relation.clone(),
                    });
                }
            }
            let (grid, block) = (config.grid, config.block);
            if let Some([x, y, z]) = kernel.required_block
                && [x, y, z] != [block.x, block.y, block.z]
            {
                return Err(crate::Error::InvalidLaunch(format!(
                    "kernel {} requires blocks of {} threads, got {}",
                    kernel.name,
                    shape(crate::Dim3::new(x, y, z)),
                    shape(block)
                )));
            }
            if kernel.one_dimensional && [grid.y, grid.z, block.y, block.z] != [1; 4] {
                return Err(crate::Error::InvalidLaunch(format!(
                    "kernel {} reads its thread index, so it needs a one-dimensional launch, got a \
                     grid of {} blocks of {} threads",
                    kernel.name,
                    shape(grid),
                    shape(block)
                )));
            }
        }
        Ok(())
    }
}

impl Compiled {
    pub(super) fn raw(&self) -> &Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        &self.raw
    }

    pub(super) fn new(device: &Device, module: &Module, entrypoint: &str) -> Result<Self> {
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
        super::module::require_simd_width(
            module
                .abi
                .as_ref()
                .map_or(&[][..], |abi| &abi.required_features),
            raw.threadExecutionWidth(),
        )?;
        let (bytes, capacity) = (
            raw.staticThreadgroupMemoryLength(),
            device.raw().maxThreadgroupMemoryLength(),
        );
        if bytes > capacity {
            return Err(Error::InvalidLaunch(format!(
                "kernel {entrypoint} uses {bytes} B of threadgroup memory; the device has {capacity} B"
            )));
        }
        if abi.is_some() {
            device.snapshot()?;
        }
        Ok(Self {
            raw,
            entrypoint: entrypoint.into(),
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
        })
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
