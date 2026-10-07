//! Metal Shading Language generation from validated kernel IR.

mod arithmetic;
mod bindings;
mod control;
mod emit;
mod expressions;
mod floating;
mod host_types;
mod interface;
mod names;
mod numeric;
mod structured;

use interface::KernelInterfaces;
use metal_oxide_ir::{Error, Module};

pub use bindings::bindings;

/// Validated IR and its shared kernel parameter bindings.
#[derive(Debug)]
pub struct Codegen<'a> {
    module: &'a Module,
    interfaces: KernelInterfaces,
}

impl<'a> Codegen<'a> {
    pub fn new(module: &'a Module) -> Result<Self, Error> {
        metal_oxide_ir::validate(module)?;
        metal_oxide_ir::prove_numerics(module)?;
        Ok(Self {
            module,
            interfaces: KernelInterfaces::new(module)?,
        })
    }

    /// Emits MSL with floating-point contraction disabled.
    ///
    /// Compile the result with safe math and precise floating-point functions.
    /// Assertions require a proof; unsupported operations are errors.
    pub fn emit(&self) -> Result<String, Error> {
        emit::ModuleEmitter::new(self.module, &self.interfaces).emit()
    }

    pub fn abi(&self) -> Result<metal_oxide_artifact::Abi, metal_oxide_artifact::Error> {
        self.interfaces.abi()
    }
}
