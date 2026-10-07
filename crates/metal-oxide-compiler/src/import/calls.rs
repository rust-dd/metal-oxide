use metal_oxide_ir as ir;
use rustc_middle::{
    mir,
    ty::{self, Instance, consts::ConstExt},
};
use rustc_span::{Span, Spanned};

use super::{Result, function::FunctionImporter};
use crate::intrinsics::Intrinsic;

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(super) fn call(
        &mut self,
        func: &mir::Operand<'tcx>,
        args: &[Spanned<mir::Operand<'tcx>>],
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Expression> {
        let ty = self.normalize_type(func.ty(&body.local_decls, self.module.tcx));
        let ty::FnDef(def, args_types) = *ty.kind() else {
            return Err((span, "indirect calls cannot be imported".into()));
        };
        let args_types = self
            .module
            .tcx
            .instantiate_bound_regions_with_erased(args_types);
        let instance = Instance::try_resolve(
            self.module.tcx,
            ty::TypingEnv::fully_monomorphized(),
            def,
            args_types,
        )
        .map_err(|_| (span, "cannot resolve concrete function".into()))?
        .ok_or((span, "cannot resolve concrete function".into()))?;
        let arguments = args
            .iter()
            .map(|a| self.operand(&a.node, a.span))
            .collect::<Result<Vec<_>>>()?;
        if let Some(intrinsic) = crate::intrinsics::builtin(self.module.tcx, def) {
            return self.intrinsic(intrinsic, args_types, arguments, span);
        }
        let function = *self
            .module
            .function_ids
            .get(&instance)
            .ok_or((span, "missing concrete function instance".into()))?;
        Ok(ir::Expression::Call {
            function,
            arguments,
        })
    }

    fn intrinsic(
        &mut self,
        intrinsic: Intrinsic,
        types: ty::GenericArgsRef<'tcx>,
        values: Vec<ir::Operand>,
        span: Span,
    ) -> Result<ir::Expression> {
        Ok(match intrinsic {
            Intrinsic::Coordinates(builtin) => {
                arguments::<0>(values, span)?;
                ir::Expression::Coordinates(builtin)
            }
            Intrinsic::SimdCoordinate(builtin) => {
                arguments::<0>(values, span)?;
                ir::Expression::SimdCoordinate(builtin)
            }
            Intrinsic::SimdSum => {
                let [value] = arguments(values, span)?;
                ir::Expression::SimdSum(value)
            }
            Intrinsic::SimdShuffle => {
                let [value, lane] = arguments(values, span)?;
                ir::Expression::SimdShuffle { value, lane }
            }
            Intrinsic::ThreadgroupAlloc => {
                arguments::<0>(values, span)?;
                let ir::Type::Scalar(element) = self.lower_type(types.type_at(0), span)? else {
                    return Err((span, "threadgroup elements must be scalars".into()));
                };
                let length = types
                    .const_at(1)
                    .try_to_target_usize(self.module.tcx)
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or((
                        span,
                        "threadgroup length must be a concrete u32-sized constant".into(),
                    ))?;
                let id = self.allocations;
                self.allocations += 1;
                ir::Expression::ThreadgroupAlloc {
                    id,
                    element,
                    length,
                }
            }
            Intrinsic::ThreadgroupBarrier => {
                arguments::<0>(values, span)?;
                ir::Expression::ThreadgroupBarrier
            }
            Intrinsic::AtomicAdd => {
                let [buffer, index, value] = arguments(values, span)?;
                ir::Expression::AtomicAdd {
                    buffer,
                    index,
                    value,
                }
            }
            Intrinsic::BufferLoad => {
                let [buffer, index] = arguments(values, span)?;
                ir::Expression::BufferLoad { buffer, index }
            }
            Intrinsic::BufferStore => {
                let [buffer, index, value] = arguments(values, span)?;
                ir::Expression::BufferStore {
                    buffer,
                    index,
                    value,
                }
            }
        })
    }
}

fn arguments<const N: usize>(values: Vec<ir::Operand>, span: Span) -> Result<[ir::Operand; N]> {
    values
        .try_into()
        .map_err(|_| (span, "invalid device builtin arguments".into()))
}
