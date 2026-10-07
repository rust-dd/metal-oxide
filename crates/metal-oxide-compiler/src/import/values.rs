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

    pub(super) fn destination(&self, place: &mir::Place<'tcx>, span: Span) -> Result<usize> {
        if !place.projection.is_empty() {
            return Err((
                span,
                "writes to projected places are not supported yet".into(),
            ));
        }
        Ok(place.local.as_usize())
    }

    pub(super) fn operand(
        &mut self,
        operand: &mir::Operand<'tcx>,
        span: Span,
    ) -> Result<ir::Operand> {
        match operand {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => {
                let field = match place.projection.as_slice() {
                    [] => None,
                    [mir::ProjectionElem::Field(field, _)] => Some(field.as_u32()),
                    _ => return Err((span, "unsupported MIR place projection".into())),
                };
                Ok(ir::Operand::Place {
                    local: place.local.as_usize(),
                    field,
                })
            }
            mir::Operand::Constant(value) => {
                let constant = self.instance.instantiate_mir_and_normalize_erasing_regions(
                    self.module.tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(self.module.tcx, value.const_),
                );
                let ty = self.lower_type(constant.ty(), span)?;
                if ty == ir::Type::Unit {
                    return Ok(ir::Operand::Constant(ir::Constant::Unit));
                }
                let bits = constant
                    .try_eval_bits(self.module.tcx, TypingEnv::fully_monomorphized())
                    .ok_or((span, "unsupported device constant".into()))?;
                let constant = match ty {
                    ir::Type::Scalar(ir::Scalar::Bool) => ir::Constant::Bool(bits != 0),
                    ir::Type::Scalar(ir::Scalar::F32) => ir::Constant::F32(bits as u32),
                    ir::Type::Scalar(ir::Scalar::U32) => ir::Constant::U32(bits as u32),
                    ir::Type::Scalar(ir::Scalar::I32) => ir::Constant::I32(bits as i32),
                    ir::Type::Scalar(ir::Scalar::U8) => ir::Constant::U8(bits as u8),
                    ir::Type::Scalar(ir::Scalar::U16) => ir::Constant::U16(bits as u16),
                    _ => return Err((span, "unsupported device constant type".into())),
                };
                Ok(ir::Operand::Constant(constant))
            }
            _ => Err((span, "unsupported MIR operand".into())),
        }
    }

    pub(super) fn expression(
        &mut self,
        value: &mir::Rvalue<'tcx>,
        span: Span,
    ) -> Result<ir::Expression> {
        match value {
            mir::Rvalue::Aggregate(kind, fields) => {
                let mir::AggregateKind::Adt(def, _, args, _, _) = kind.as_ref() else {
                    return Err((span, "unsupported MIR aggregate".into()));
                };
                let ty = self.normalize_type(
                    self.module
                        .tcx
                        .type_of(*def)
                        .instantiate(self.module.tcx, args)
                        .skip_norm_wip(),
                );
                let ty = self.lower_type(ty, span)?;
                let fields = fields
                    .iter()
                    .map(|value| self.operand(value, span))
                    .collect::<Result<Vec<_>>>()?;
                match ty {
                    ir::Type::Record(ty) => Ok(ir::Expression::Record { ty, fields }),
                    ir::Type::Dim3 => Ok(ir::Expression::Dim3(
                        fields
                            .try_into()
                            .map_err(|_| (span, "invalid Dim3 aggregate".into()))?,
                    )),
                    _ => Err((span, "unsupported MIR aggregate type".into())),
                }
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
