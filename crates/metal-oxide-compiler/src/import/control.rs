use metal_oxide_ir as ir;
use rustc_middle::mir::{
    self,
    visit::{PlaceContext, Visitor},
};
use rustc_span::Span;

use super::{Context, Result};

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

impl<'tcx> Context<'_, 'tcx> {
    pub(super) fn switch(
        &self,
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
        let ty = self.ty(
            self.normalize_type(discriminant.ty(&body.local_decls, self.tcx)),
            span,
        )?;
        if ty == ir::Type::Scalar(ir::Scalar::Bool) {
            let [(value, target)] = cases.as_slice() else {
                return Err((span, "invalid bool switch cases".into()));
            };
            let (then_block, else_block) = match value {
                0 => (otherwise, target.as_usize()),
                1 => (target.as_usize(), otherwise),
                _ => return Err((span, "invalid bool switch value".into())),
            };
            return Ok(ir::Terminator::Branch {
                condition: operand,
                then_block,
                else_block,
            });
        }
        let cases = cases
            .into_iter()
            .map(|(value, target)| {
                let value = match ty {
                    ir::Type::Scalar(ir::Scalar::U32) => ir::Constant::U32(value as u32),
                    ir::Type::Scalar(ir::Scalar::I32) => ir::Constant::I32(value as i32),
                    ir::Type::Scalar(ir::Scalar::U8) => ir::Constant::U8(value as u8),
                    ir::Type::Scalar(ir::Scalar::U16) => ir::Constant::U16(value as u16),
                    _ => return Err((span, "unsupported switch discriminant type".into())),
                };
                Ok((value, target.as_usize()))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ir::Terminator::Switch {
            discriminant: operand,
            cases,
            otherwise,
        })
    }
}
