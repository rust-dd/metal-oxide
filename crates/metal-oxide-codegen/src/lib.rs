//! Metal Shading Language generation from validated kernel IR.

mod arithmetic;
mod atomics;
mod bindings;
mod checks;
mod control;
mod emit;
mod expressions;
mod floating;
mod gates;
mod host_types;
mod integer;
mod interface;
mod layout;
mod log;
mod names;
mod numeric;
mod simd;
mod structured;
mod unused;

use checks::CheckTable;
use interface::KernelInterfaces;
use metal_oxide_artifact::MathMode;
use metal_oxide_ir::{Error, FunctionAnalysis, Module};

pub use bindings::bindings;

/// Validated IR and its shared kernel parameter bindings.
#[derive(Debug)]
pub struct Codegen<'a> {
    module: &'a Module,
    interfaces: KernelInterfaces,
    analyses: Vec<FunctionAnalysis>,
    checks: CheckTable,
    math_mode: MathMode,
}

impl<'a> Codegen<'a> {
    pub fn new(module: &'a Module) -> Result<Self, Error> {
        let analyses = metal_oxide_ir::validate(module)?;
        Ok(Self {
            module,
            interfaces: KernelInterfaces::new(module, &analyses)?,
            checks: CheckTable::new(module, &analyses),
            analyses,
            math_mode: MathMode::default(),
        })
    }

    /// Selects the floating-point contract for both MSL and artifact metadata.
    pub fn with_math_mode(mut self, math_mode: MathMode) -> Self {
        self.math_mode = math_mode;
        self
    }

    /// Emits MSL with explicit numerical pragmas and math function namespaces.
    ///
    /// The default is precise. Offline compilation uses [`MathMode::metal_flags`].
    /// Unproved assertions become runtime checks; unsupported operations are errors.
    pub fn emit(&self) -> Result<String, Error> {
        emit::ModuleEmitter::new(
            self.module,
            &self.interfaces,
            &self.analyses,
            &self.checks,
            self.math_mode,
        )
        .emit()
    }

    /// The runtime checks the emitted MSL reports, indexed by check id.
    pub fn checks(&self) -> &[metal_oxide_artifact::Check] {
        &self.checks.entries
    }

    pub fn abi(&self) -> Result<metal_oxide_artifact::Abi, metal_oxide_artifact::Error> {
        self.interfaces.abi(self.math_mode)
    }
}
