use metal_oxide_ir as ir;
use rustc_middle::{
    mir,
    ty::{self, Instance, consts::ConstExt},
};
use rustc_span::{Span, Spanned};

use super::{Result, function::FunctionImporter};
use crate::intrinsics::Intrinsic;

/// A resolved call target.
pub(super) struct Callee<'tcx> {
    pub instance: Instance<'tcx>,
    pub definition: rustc_hir::def_id::DefId,
    /// The generic arguments of the called definition, before resolution.
    pub arguments: ty::GenericArgsRef<'tcx>,
    pub intrinsic: Option<Intrinsic>,
}

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(super) fn callee(
        &self,
        func: &mir::Operand<'tcx>,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<Callee<'tcx>> {
        let tcx = self.module.tcx;
        let ty = self.normalize_type(func.ty(&body.local_decls, tcx));
        let ty::FnDef(definition, arguments) = *ty.kind() else {
            return Err((span, "indirect calls cannot be imported".into()));
        };
        let arguments = tcx.instantiate_bound_regions_with_erased(arguments);
        let instance = Instance::try_resolve(
            tcx,
            ty::TypingEnv::fully_monomorphized(),
            definition,
            arguments,
        )
        .ok()
        .flatten()
        .ok_or((span, "cannot resolve concrete function".into()))?;
        let intrinsic = crate::intrinsics::builtin(tcx, definition)
            .or_else(|| crate::intrinsics::builtin(tcx, instance.def_id()));
        Ok(Callee {
            instance,
            definition,
            arguments,
            intrinsic,
        })
    }

    pub(super) fn call(
        &mut self,
        func: &mir::Operand<'tcx>,
        args: &[Spanned<mir::Operand<'tcx>>],
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Expression> {
        let callee = self.callee(func, body, span)?;
        if let Some(Intrinsic::Log) = callee.intrinsic {
            return self.log(args, callee.arguments, span);
        }
        let arguments = args
            .iter()
            .map(|a| self.operand(&a.node, a.span))
            .collect::<Result<Vec<_>>>()?;
        if let Some(intrinsic) = callee.intrinsic {
            return self.intrinsic(intrinsic, callee.arguments, arguments, span);
        }
        let mut types = Vec::with_capacity(args.len());
        for argument in args {
            types.push(
                self.argument_type(&argument.node, &[], argument.span)?
                    .expect("inferred references are known"),
            );
        }
        Ok(ir::Expression::Call {
            function: self.module.function(callee.instance, types, false)?,
            arguments,
        })
    }

    /// `log::write(format, tuple)`: the literal format and one operand per tuple field.
    fn log(
        &mut self,
        args: &[Spanned<mir::Operand<'tcx>>],
        types: ty::GenericArgsRef<'tcx>,
        span: Span,
    ) -> Result<ir::Expression> {
        let format = super::panics::literal(self.module.tcx, &args[0].node)
            .ok_or((span, "device log formats must be string literals".into()))?;
        let ty::Tuple(fields) = types.type_at(0).kind() else {
            return Err((span, "device log arguments must be a tuple".into()));
        };
        let arguments = match self.operand(&args[1].node, args[1].span)? {
            ir::Operand::Place(place) => (0..fields.len() as u32)
                .map(|field| {
                    let mut place = place.clone();
                    place.projection.push(ir::Projection::Field(field));
                    ir::Operand::Place(place)
                })
                .collect(),
            ir::Operand::AggregateConstant { fields, .. } => fields,
            ir::Operand::Constant(ir::Constant::Unit) => Vec::new(),
            ir::Operand::Constant(_) => {
                return Err((span, "device log arguments must be a tuple".into()));
            }
        };
        Ok(ir::Expression::Log { format, arguments })
    }

    fn intrinsic(
        &mut self,
        intrinsic: Intrinsic,
        types: ty::GenericArgsRef<'tcx>,
        values: Vec<ir::Operand>,
        span: Span,
    ) -> Result<ir::Expression> {
        Ok(match intrinsic {
            Intrinsic::Math(op) => {
                if values.len() != op.arity() {
                    return Err((span, "invalid math intrinsic arity".into()));
                }
                ir::Expression::Math {
                    op,
                    arguments: values,
                }
            }
            Intrinsic::Integer(op) => {
                if values.len() != op.arity() {
                    return Err((span, "invalid integer intrinsic arity".into()));
                }
                ir::Expression::Integer {
                    op,
                    arguments: values,
                }
            }
            Intrinsic::Bitcast(to) => {
                let [value] = arguments(values, span)?;
                ir::Expression::Bitcast(value, to)
            }
            Intrinsic::FloatConvert(to) => {
                let [value] = arguments(values, span)?;
                ir::Expression::Cast(value, to)
            }
            Intrinsic::Log => unreachable!("log calls are imported before their operands"),
            Intrinsic::UncheckedFloatToInt => {
                let [value] = arguments(values, span)?;
                let ir::Type::Scalar(to) = self.lower_type(types.type_at(0), span)? else {
                    return Err((span, "unchecked float conversion requires a scalar".into()));
                };
                ir::Expression::UncheckedCast(value, to)
            }
            Intrinsic::WrappingShift(op) => {
                let [value, count] = arguments(values, span)?;
                ir::Expression::Binary(op, value, count)
            }
            Intrinsic::Coordinates(builtin) => {
                arguments::<0>(values, span)?;
                ir::Expression::Coordinates(builtin)
            }
            Intrinsic::SimdCoordinate(builtin) => {
                arguments::<0>(values, span)?;
                ir::Expression::SimdCoordinate(builtin)
            }
            Intrinsic::Simd(op) => {
                if values.len() != op.arity() {
                    return Err((span, "invalid SIMD intrinsic arity".into()));
                }
                ir::Expression::Simd {
                    op,
                    arguments: values,
                }
            }
            Intrinsic::ThreadgroupAlloc(access) => {
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
                    access,
                }
            }
            Intrinsic::ThreadgroupBarrier => {
                arguments::<0>(values, span)?;
                ir::Expression::ThreadgroupBarrier
            }
            Intrinsic::Uninit => {
                arguments::<0>(values, span)?;
                let array = ty::Ty::new_array_with_const_len(
                    self.module.tcx,
                    types.type_at(0),
                    types.const_at(1),
                );
                ir::Expression::Uninit(self.lower_type(array, span)?)
            }
            Intrinsic::Atomic(op) => {
                if values.len() != op.arity() + 2 {
                    return Err((span, "invalid atomic intrinsic arity".into()));
                }
                ir::Expression::Atomic {
                    op,
                    arguments: values,
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
            Intrinsic::SliceLen => {
                let [slice] = arguments(values, span)?;
                ir::Expression::Len(slice)
            }
            Intrinsic::Identity => {
                let [value] = arguments(values, span)?;
                ir::Expression::Use(value)
            }
            Intrinsic::Subslice | Intrinsic::Buffer(_) | Intrinsic::SliceIterator(_) => {
                unreachable!("subslices, buffer queries and slice iterators are expanded in place")
            }
        })
    }
}

fn arguments<const N: usize>(values: Vec<ir::Operand>, span: Span) -> Result<[ir::Operand; N]> {
    values
        .try_into()
        .map_err(|_| (span, "invalid device builtin arguments".into()))
}
