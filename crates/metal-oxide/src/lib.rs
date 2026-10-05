//! Typed buffers and synchronous Metal compute on Apple Silicon.

mod error;
mod launch;
mod scalar;

pub use error::{Error, Result};
pub use launch::{Dim3, DynamicLaunchConfig, LaunchConfig};
pub use scalar::{GpuAtomic, GpuScalar};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod metal;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use metal::{Argument, Buffer, Device, Module, Pipeline};
