use metal_oxide_ir as ir;
use rustc_middle::{mir, ty};
use rustc_span::{Span, Spanned};

use super::{Result, function::FunctionImporter};

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(super) fn atomic(
        &mut self,
        op: ir::AtomicOp,
        args: &[Spanned<mir::Operand<'tcx>>],
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Expression> {
        let operands = 2 + op.arity();
        let orders = if op == ir::AtomicOp::CompareExchangeWeak {
            2
        } else {
            1
        };
        if args.len() != operands + orders {
            return Err((span, "invalid atomic intrinsic arity".into()));
        }
        let order = self.atomic_order(&args[operands], body)?;
        let failure = if orders == 2 {
            self.atomic_order(&args[operands + 1], body)?
        } else {
            ir::AtomicOrdering::Relaxed
        };
        op.validate_order(order, failure)
            .map_err(|error| (span, error.into()))?;
        Ok(ir::Expression::Atomic {
            op,
            order,
            failure,
            arguments: args[..operands]
                .iter()
                .map(|arg| self.operand(&arg.node, arg.span))
                .collect::<Result<_>>()?,
        })
    }

    fn atomic_order(
        &mut self,
        argument: &Spanned<mir::Operand<'tcx>>,
        body: &mir::Body<'tcx>,
    ) -> Result<ir::AtomicOrdering> {
        let tcx = self.module.tcx;
        let rust = self.normalize_type(argument.node.ty(&body.local_decls, tcx));
        let ty::Adt(definition, _) = rust.kind() else {
            return Err((
                argument.span,
                "atomic ordering requires core::sync::atomic::Ordering".into(),
            ));
        };
        let variant = self
            .order_variant(&argument.node, body, argument.span, &mut Vec::new())?
            .ok_or((
                argument.span,
                "atomic ordering must be a compile-time constant".into(),
            ))?;
        let variant = &definition.variants()[variant];
        Ok(match variant.name.as_str() {
            "Relaxed" => ir::AtomicOrdering::Relaxed,
            "Acquire" => ir::AtomicOrdering::Acquire,
            "Release" => ir::AtomicOrdering::Release,
            "AcqRel" => ir::AtomicOrdering::AcqRel,
            "SeqCst" => ir::AtomicOrdering::SeqCst,
            _ => return Err((argument.span, "unsupported atomic ordering".into())),
        })
    }
    fn order_variant(
        &mut self,
        value: &mir::Operand<'tcx>,
        body: &mir::Body<'tcx>,
        span: Span,
        visiting: &mut Vec<mir::Local>,
    ) -> Result<Option<rustc_abi::VariantIdx>> {
        if matches!(value, mir::Operand::Constant(_)) {
            return Ok(match self.operand(value, span)? {
                ir::Operand::AggregateConstant { fields, .. } => match fields.first() {
                    Some(ir::Operand::Constant(ir::Constant::U8(tag))) => {
                        Some(rustc_abi::VariantIdx::from_u32(u32::from(*tag)))
                    }
                    _ => None,
                },
                _ => None,
            });
        }
        let Some(place) = value.place().filter(|p| p.projection.is_empty()) else {
            return Ok(None);
        };
        if place.local.as_usize() <= body.arg_count || visiting.contains(&place.local) {
            return Ok(None);
        }
        visiting.push(place.local);
        let mut assignment = None;
        for block in body.basic_blocks.iter() {
            if matches!(&block.terminator().kind, mir::TerminatorKind::Call { destination, .. } if destination.local == place.local)
            {
                return Ok(None);
            }
            for statement in &block.statements {
                if let Some((target, rhs)) = statement.kind.as_assign() {
                    if matches!(rhs, mir::Rvalue::Ref(_, _, borrowed) | mir::Rvalue::RawPtr(_, borrowed) if borrowed.local == place.local)
                    {
                        return Ok(None);
                    }
                    if target.local == place.local {
                        if !target.projection.is_empty() || assignment.is_some() {
                            return Ok(None);
                        }
                        assignment = Some(rhs);
                    }
                }
            }
        }
        match assignment {
            Some(mir::Rvalue::Aggregate(kind, fields)) if fields.is_empty() => match **kind {
                mir::AggregateKind::Adt(_, variant, ..) => Ok(Some(variant)),
                _ => Ok(None),
            },
            Some(mir::Rvalue::Use(value, _)) => self.order_variant(value, body, span, visiting),
            _ => Ok(None),
        }
    }
}
