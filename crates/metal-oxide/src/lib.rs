//! Typed buffers and synchronous Metal compute on Apple Silicon.

mod dispatch;
mod error;
mod scalar;

pub use dispatch::Dispatch1d;
pub use error::{Error, Result};
pub use scalar::GpuScalar;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod metal;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use metal::{Argument, Buffer, Device, Module, Pipeline};
