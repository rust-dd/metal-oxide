use metal_oxide_artifact::{Comparison, Contract, Term};
use metal_oxide_ir as ir;
use rustc_middle::{mir, ty};

use super::{Result, function::FunctionImporter};

impl<'tcx> FunctionImporter<'_, 'tcx> {
    /// A kernel's `requires` relations, over its buffer lengths and integer parameters.
    pub(super) fn requirements(&self, body: &mir::Body<'tcx>) -> Result<Vec<Contract>> {
        let operand = |name: &str| {
            let index = (0..body.arg_count).find(|&index| {
                crate::collect::parameter_name(body, index).as_deref() == Some(name)
            })?;
            let operand = match self.parameters[index] {
                ir::Type::Buffer {
                    address_space: ir::AddressSpace::Device,
                    ..
                } => metal_oxide_artifact::Operand::Buffer,
                ir::Type::Scalar(scalar) if scalar.is_integer() => {
                    metal_oxide_artifact::Operand::Integer
                }
                _ => return None,
            };
            Some((index, operand))
        };
        crate::collect::requires(self.module.tcx, self.instance.def_id())?
            .into_iter()
            .map(|(relation, span)| {
                Contract::parse(&relation, operand).map_err(|error| (span, error.0))
            })
            .collect()
    }

    /// What the types of a helper's integer parameters guarantee at its entry: `NonZero`
    /// excludes zero, which proves divisions by it, as in core's `Div<NonZero<T>>`.
    pub(super) fn invariants(&self, body: &mir::Body<'tcx>) -> Vec<Contract> {
        let tcx = self.module.tcx;
        (0..body.arg_count)
            .filter(|&index| {
                matches!(self.parameters[index], ir::Type::Scalar(scalar) if scalar.is_integer())
            })
            .filter(|&index| {
                let local = mir::Local::from_usize(index + 1);
                let ty = self.normalize_type(body.local_decls[local].ty);
                tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty))
                    .is_ok_and(|layout| match layout.backend_repr {
                        rustc_abi::BackendRepr::Scalar(scalar) => {
                            let range = scalar.valid_range(&tcx);
                            range.start == 1 && range.end == scalar.size(&tcx).unsigned_int_max()
                        }
                        _ => false,
                    })
            })
            .map(|index| Contract {
                relation: format!("parameter {index} is nonzero"),
                left: Term::Value(index),
                comparison: Comparison::Ne,
                right: Term::Literal(0),
            })
            .collect()
    }
}
