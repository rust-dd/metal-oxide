use metal_oxide_ir as ir;
use rustc_middle::{
    mir,
    ty::{self, EarlyBinder, Ty, TypingEnv},
};
use rustc_span::{Span, Symbol};

use super::{Context, Result};

impl<'tcx> Context<'_, 'tcx> {
    pub(super) fn normalize_type(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(self.tcx, ty),
        )
    }

    pub(super) fn ty(&self, ty: Ty<'tcx>, span: Span) -> Result<ir::Type> {
        let scalar = |ty: Ty<'tcx>| match ty.kind() {
            ty::Bool => Some(ir::Scalar::Bool),
            ty::Float(ty::FloatTy::F32) => Some(ir::Scalar::F32),
            ty::Uint(ty::UintTy::U32) => Some(ir::Scalar::U32),
            ty::Int(ty::IntTy::I32) => Some(ir::Scalar::I32),
            _ => None,
        };
        if let Some(s) = scalar(ty) {
            return Ok(ir::Type::Scalar(s));
        }
        if ty.is_unit() {
            return Ok(ir::Type::Unit);
        }
        if ty.is_never() {
            return Ok(ir::Type::Never);
        }
        if let ty::Tuple(types) = ty.kind()
            && types.len() == 2
            && types[1].is_bool()
            && let Some(s @ (ir::Scalar::U32 | ir::Scalar::I32)) = scalar(types[0])
        {
            return Ok(ir::Type::Checked(s));
        }
        if let ty::Adt(definition, args) = ty.kind() {
            let marker =
                |name| self.tcx.get_diagnostic_item(Symbol::intern(name)) == Some(definition.did());
            if marker("metal_oxide_dim3") {
                return Ok(ir::Type::Dim3);
            }
            for (name, access) in [
                ("metal_oxide_read_buffer", ir::Access::Read),
                ("metal_oxide_write_buffer", ir::Access::Write),
                ("metal_oxide_atomic_buffer", ir::Access::Atomic),
                ("metal_oxide_threadgroup_buffer", ir::Access::ReadWrite),
            ] {
                if marker(name)
                    && let Some(element @ (ir::Scalar::F32 | ir::Scalar::U32 | ir::Scalar::I32)) =
                        scalar(args.type_at(0))
                {
                    return Ok(ir::Type::Buffer {
                        element,
                        access,
                        address_space: if access == ir::Access::ReadWrite {
                            ir::AddressSpace::Threadgroup
                        } else {
                            ir::AddressSpace::Device
                        },
                    });
                }
            }
        }
        Err((span, format!("unsupported device type: {ty}")))
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

    pub(super) fn operand(&self, operand: &mir::Operand<'tcx>, span: Span) -> Result<ir::Operand> {
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
                    self.tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(self.tcx, value.const_),
                );
                let ty = self.ty(constant.ty(), span)?;
                if ty == ir::Type::Unit {
                    return Ok(ir::Operand::Constant(ir::Constant::Unit));
                }
                let bits = constant
                    .try_eval_bits(self.tcx, TypingEnv::fully_monomorphized())
                    .ok_or((span, "unsupported device constant".into()))?;
                let constant = match ty {
                    ir::Type::Scalar(ir::Scalar::Bool) => ir::Constant::Bool(bits != 0),
                    ir::Type::Scalar(ir::Scalar::F32) => ir::Constant::F32(bits as u32),
                    ir::Type::Scalar(ir::Scalar::U32) => ir::Constant::U32(bits as u32),
                    ir::Type::Scalar(ir::Scalar::I32) => ir::Constant::I32(bits as i32),
                    _ => return Err((span, "unsupported device constant type".into())),
                };
                Ok(ir::Operand::Constant(constant))
            }
            _ => Err((span, "unsupported MIR operand".into())),
        }
    }

    pub(super) fn expression(
        &self,
        value: &mir::Rvalue<'tcx>,
        span: Span,
    ) -> Result<ir::Expression> {
        let operand = |v| self.operand(v, span);
        match value {
            mir::Rvalue::Use(v, _) => Ok(ir::Expression::Use(operand(v)?)),
            mir::Rvalue::BinaryOp(op, values) => Ok(ir::Expression::Binary(
                binary(*op, span)?,
                operand(&values.0)?,
                operand(&values.1)?,
            )),
            mir::Rvalue::UnaryOp(op, v) => {
                let op = match op {
                    mir::UnOp::Neg => ir::UnaryOp::Neg,
                    mir::UnOp::Not => ir::UnaryOp::Not,
                    _ => return Err((span, "unsupported MIR unary operation".into())),
                };
                Ok(ir::Expression::Unary(op, operand(v)?))
            }
            mir::Rvalue::Cast(
                mir::CastKind::IntToInt
                | mir::CastKind::IntToFloat
                | mir::CastKind::FloatToInt
                | mir::CastKind::FloatToFloat,
                v,
                to,
            ) => {
                let ir::Type::Scalar(to) = self.ty(self.normalize_type(*to), span)? else {
                    return Err((span, "unsupported cast destination".into()));
                };
                Ok(ir::Expression::Cast(operand(v)?, to))
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
