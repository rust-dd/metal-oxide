use metal_oxide_ir as ir;
use rustc_middle::{
    mir,
    ty::{EarlyBinder, Ty, TypingEnv},
};
use rustc_span::Span;

use super::{Result, function::FunctionImporter};

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(super) fn normalize_type(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.module.tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(self.module.tcx, ty),
        )
    }

    pub(super) fn lower_type(&mut self, ty: Ty<'tcx>, span: Span) -> Result<ir::Type> {
        let ty = self.normalize_type(ty);
        self.module.types.lower(ty, span)
    }

    pub(super) fn operand(
        &mut self,
        operand: &mir::Operand<'tcx>,
        span: Span,
    ) -> Result<ir::Operand> {
        match operand {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => {
                if place.projection.is_empty()
                    && let Some(value) = self.constants[place.local.as_usize()]
                {
                    return Ok(ir::Operand::Constant(value));
                }
                Ok(ir::Operand::Place(self.place(place, span)?))
            }
            mir::Operand::Constant(value) => {
                let constant = self.instance.instantiate_mir_and_normalize_erasing_regions(
                    self.module.tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(self.module.tcx, value.const_),
                );
                if let Some(value) = super::constants::literal(self.module.tcx, constant) {
                    return Ok(ir::Operand::Constant(value));
                }
                self.owned_literal(constant, span)
            }
            _ => Err((span, "unsupported MIR operand".into())),
        }
    }

    pub(super) fn expression(
        &mut self,
        value: &mir::Rvalue<'tcx>,
        destination: Ty<'tcx>,
        span: Span,
    ) -> Result<ir::Expression> {
        match value {
            mir::Rvalue::Aggregate(_, fields) => {
                let ty = self.lower_type(destination, span)?;
                let fields = fields
                    .iter()
                    .map(|value| self.operand(value, span))
                    .collect::<Result<Vec<_>>>()?;
                match ty {
                    ir::Type::Checked(scalar) => {
                        let [value, overflow] = fields
                            .try_into()
                            .map_err(|_| (span, "invalid checked tuple".into()))?;
                        Ok(ir::Expression::Checked {
                            scalar,
                            value,
                            overflow,
                        })
                    }
                    ir::Type::Aggregate(ty) => Ok(ir::Expression::Aggregate { ty, fields }),
                    ir::Type::Dim3 => Ok(ir::Expression::Dim3(
                        fields
                            .try_into()
                            .map_err(|_| (span, "invalid Dim3 aggregate".into()))?,
                    )),
                    _ => Err((span, "unsupported MIR aggregate type".into())),
                }
            }
            mir::Rvalue::Repeat(value, _) => {
                let ir::Type::Aggregate(ty) = self.lower_type(destination, span)? else {
                    unreachable!()
                };
                let count = self.module.types.table().get(ty).unwrap().len();
                Ok(ir::Expression::Aggregate {
                    ty,
                    fields: vec![self.operand(value, span)?; count],
                })
            }
            mir::Rvalue::Use(v, _) => Ok(ir::Expression::Use(self.operand(v, span)?)),
            mir::Rvalue::BinaryOp(op, values) => Ok(ir::Expression::Binary(
                binary(*op, span)?,
                self.operand(&values.0, span)?,
                self.operand(&values.1, span)?,
            )),
            mir::Rvalue::UnaryOp(op, v) => {
                let op = match op {
                    mir::UnOp::Neg => ir::UnaryOp::Neg,
                    mir::UnOp::Not => ir::UnaryOp::Not,
                    _ => return Err((span, "unsupported MIR unary operation".into())),
                };
                Ok(ir::Expression::Unary(op, self.operand(v, span)?))
            }
            mir::Rvalue::Cast(
                mir::CastKind::IntToInt
                | mir::CastKind::IntToFloat
                | mir::CastKind::FloatToInt
                | mir::CastKind::FloatToFloat,
                v,
                to,
            ) => {
                let ir::Type::Scalar(to) = self.lower_type(*to, span)? else {
                    return Err((span, "unsupported cast destination".into()));
                };
                Ok(ir::Expression::Cast(self.operand(v, span)?, to))
            }
            other => Err((span, format!("unsupported MIR rvalue: {other:?}"))),
        }
    }
}

fn binary(op: mir::BinOp, span: Span) -> Result<ir::BinaryOp> {
    Ok(match op {
        mir::BinOp::Add => ir::BinaryOp::Add,
        mir::BinOp::Sub => ir::BinaryOp::Sub,
        mir::BinOp::Mul => ir::BinaryOp::Mul,
        mir::BinOp::Div => ir::BinaryOp::Div,
        mir::BinOp::Rem => ir::BinaryOp::Rem,
        mir::BinOp::BitAnd => ir::BinaryOp::BitAnd,
        mir::BinOp::BitOr => ir::BinaryOp::BitOr,
        mir::BinOp::BitXor => ir::BinaryOp::BitXor,
        mir::BinOp::Shl => ir::BinaryOp::Shl,
        mir::BinOp::Shr => ir::BinaryOp::Shr,
        mir::BinOp::Eq => ir::BinaryOp::Eq,
        mir::BinOp::Ne => ir::BinaryOp::Ne,
        mir::BinOp::Lt => ir::BinaryOp::Lt,
        mir::BinOp::Le => ir::BinaryOp::Le,
        mir::BinOp::Gt => ir::BinaryOp::Gt,
        mir::BinOp::Ge => ir::BinaryOp::Ge,
        mir::BinOp::AddWithOverflow => ir::BinaryOp::AddWithOverflow,
        mir::BinOp::SubWithOverflow => ir::BinaryOp::SubWithOverflow,
        mir::BinOp::MulWithOverflow => ir::BinaryOp::MulWithOverflow,
        _ => return Err((span, format!("unsupported MIR binary operation: {op:?}"))),
    })
}
