//! Typed buffers and Metal compute on Apple Silicon.

mod error;
mod launch;
mod scalar;
mod value;

pub use error::{Error, KernelFailure, Result};
pub use half::f16 as F16;
pub use launch::{Dim3, DynamicLaunchConfig, LaunchConfig};
pub use metal_oxide_artifact::{Layout, LayoutKind};
pub use scalar::GpuAtomic;
pub use value::GpuValue;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod metal;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use metal::{
    Argument, Batch, Buffer, Device, Launch, Module, Operation, Pending, Pipeline, Read,
    ReadArgument, Sequence, Timed, TimedPending, Write, WriteArgument, sequence,
};
