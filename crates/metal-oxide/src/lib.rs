//! Typed buffers and Metal compute on Apple Silicon.

mod error;
mod launch;
mod scalar;
mod value;

pub use error::{Error, KernelFailure, Result};
pub use half::{bf16 as BF16, f16 as F16};
pub use launch::{Dim3, DynamicLaunchConfig, LaunchConfig};
pub use metal_oxide_artifact::{Layout, LayoutKind};
/// The device API that kernels use.
pub use metal_oxide_device as device;
pub use metal_oxide_kernels::{kernel, kernels};
pub use scalar::GpuAtomic;
pub use value::GpuValue;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod metal;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use metal::{
    Argument, Batch, BatchOperation, Buffer, Device, Launch, Library, Module, Operation, Pending,
    Pipeline, Read, ReadArgument, Sequence, Timed, TimedPending, Write, WriteArgument, sequence,
};
