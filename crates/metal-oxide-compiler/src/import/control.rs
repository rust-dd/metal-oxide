use metal_oxide_ir as ir;
use rustc_middle::mir::{
    self,
    visit::{PlaceContext, Visitor},
};
use rustc_span::Span;

use super::{Result, function::FunctionImporter};

pub(super) fn used_locals(body: &mir::Body<'_>) -> Vec<bool> {
    struct Used(Vec<bool>);
    impl<'tcx> Visitor<'tcx> for Used {
        fn visit_local(&mut self, local: mir::Local, context: PlaceContext, _: mir::Location) {
            if !matches!(context, PlaceContext::NonUse(_)) {
                self.0[local.as_usize()] = true;
            }
        }
    }
    let mut used = Used(vec![false; body.local_decls.len()]);
    used.0[..=body.arg_count].fill(true);
    used.visit_body(body);
    used.0
}

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(super) fn switch(
        &mut self,
        discriminant: &mir::Operand<'tcx>,
        targets: &mir::SwitchTargets,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Terminator> {
        let cases = targets.iter().collect::<Vec<_>>();
        let otherwise = targets.otherwise().as_usize();
        if cases.is_empty() {
            return Ok(ir::Terminator::Goto(otherwise));
        }
        let operand = self.operand(discriminant, span)?;
        // A local's imported type also covers `Option` flags, whose Rust type is `isize`.
        let ty = match discriminant.place().and_then(|place| place.as_local()) {
            Some(local) => self.locals[local.as_usize()],
            None => self.lower_type(discriminant.ty(&body.local_decls, self.module.tcx), span)?,
        };
        if ty == ir::Type::Scalar(ir::Scalar::Bool) {
            let (then_block, else_block) = match cases.as_slice() {
                [(0, target)] => (otherwise, target.as_usize()),
                [(1, target)] => (target.as_usize(), otherwise),
                [(0, no), (1, yes)] | [(1, yes), (0, no)] => (yes.as_usize(), no.as_usize()),
                _ => return Err((span, "invalid bool switch cases".into())),
            };
            // Constant conditions (often associated consts) leave the untaken arm unreachable.
            if let ir::Operand::Constant(ir::Constant::Bool(value)) = operand {
                return Ok(ir::Terminator::Goto(if value {
                    then_block
                } else {
                    else_block
                }));
            }
            return Ok(ir::Terminator::Branch {
                condition: operand,
                then_block,
                else_block,
            });
        }
        let ir::Type::Scalar(scalar) = ty else {
            return Err((span, "unsupported switch discriminant type".into()));
        };
        let cases = cases
            .into_iter()
            .map(|(value, target)| {
                let value = integer(scalar, value)
                    .ok_or((span, "unsupported switch discriminant type".into()))?;
                Ok((value, target.as_usize()))
            })
            .collect::<Result<Vec<_>>>()?;
        if let ir::Operand::Constant(value) = operand {
            let target = cases.iter().find(|(case, _)| *case == value);
            return Ok(ir::Terminator::Goto(
                target.map_or(otherwise, |&(_, target)| target),
            ));
        }
        Ok(ir::Terminator::Switch {
            discriminant: operand,
            cases,
            otherwise,
        })
    }
}

/// The integer constant of `scalar` holding the low bits of `value`.
pub(super) fn integer(scalar: ir::Scalar, value: u128) -> Option<ir::Constant> {
    Some(match scalar {
        ir::Scalar::U32 => ir::Constant::U32(value as u32),
        ir::Scalar::Usize => ir::Constant::Usize(value as u64),
        ir::Scalar::U64 => ir::Constant::U64(value as u64),
        ir::Scalar::I32 => ir::Constant::I32(value as i32),
        ir::Scalar::U8 => ir::Constant::U8(value as u8),
        ir::Scalar::U16 => ir::Constant::U16(value as u16),
        ir::Scalar::I8 => ir::Constant::I8(value as i8),
        ir::Scalar::I16 => ir::Constant::I16(value as i16),
        _ => return None,
    })
}
