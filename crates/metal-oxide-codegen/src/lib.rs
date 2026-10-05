//! Metal Shading Language generation from validated kernel IR.

mod abi;
mod bindings;
mod control;
mod emit;
mod expressions;
mod names;
mod numeric;

use metal_oxide_ir::{Error, Module};

pub use abi::abi;
pub use bindings::bindings;

/// Emits MSL with floating-point contraction disabled.
///
/// Compile the result with safe math and precise floating-point functions.
/// Enabled assertions and unsupported numerical/control-flow operations are errors.
pub fn emit(module: &Module) -> Result<String, Error> {
    metal_oxide_ir::validate(module)?;
    emit::module(module)
}
