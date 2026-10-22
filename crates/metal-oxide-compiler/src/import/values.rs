use metal_oxide_ir as ir;
use rustc_middle::{
    mir,
    ty::{EarlyBinder, Ty, TypingEnv, adjustment::PointerCoercion},
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
            mir::Operand::RuntimeChecks(checks) => Ok(ir::Operand::Constant(ir::Constant::Bool(
                checks.value(self.module.tcx.sess),
            ))),
        }
    }

    pub(super) fn expression(
        &mut self,
        value: &mir::Rvalue<'tcx>,
        destination: ir::Type,
        span: Span,
    ) -> Result<ir::Expression> {
        match value {
            mir::Rvalue::Aggregate(kind, fields)
                if fields.is_empty()
                    && destination == ir::Type::Scalar(ir::Scalar::I8)
                    && matches!(**kind, mir::AggregateKind::Adt(definition, ..) if self.module.tcx.def_path_str(definition) == "core::cmp::Ordering") =>
            {
                let mir::AggregateKind::Adt(definition, variant, ..) = **kind else {
                    unreachable!()
                };
                let value = self
                    .module
                    .tcx
                    .adt_def(definition)
                    .discriminants(self.module.tcx)
                    .find(|(index, _)| *index == variant)
                    .unwrap()
                    .1
                    .val;
                Ok(ir::Expression::Use(ir::Operand::Constant(
                    ir::Constant::I8(value as i8),
                )))
            }
            mir::Rvalue::Aggregate(kind, fields)
                if matches!(**kind, mir::AggregateKind::Adt(definition, ..)
                    if self.module.tcx.adt_def(definition).is_enum()) =>
            {
                let (mir::AggregateKind::Adt(_, variant, ..), ir::Type::Aggregate(ty)) =
                    (&**kind, destination)
                else {
                    return Err((span, "an enum value requires an enum place".into()));
                };
                Ok(ir::Expression::Variant {
                    ty,
                    variant: variant.as_u32(),
                    fields: fields
                        .iter()
                        .map(|value| self.operand(value, span))
                        .collect::<Result<Vec<_>>>()?,
                })
            }
            // A raw-pointer aggregate joins a slice's data and length; a restricted value, such
            // as a `NonNull`, is the value it holds.
            mir::Rvalue::Aggregate(kind, fields) if self.is_wrapping(kind) => {
                let data = &fields[rustc_abi::FieldIdx::ZERO];
                match self.module.types.table().pointer(destination) {
                    Some(pointer) if pointer.slice && fields.len() == 2 => {
                        Ok(ir::Expression::SliceFromParts {
                            data: self.operand(data, span)?,
                            length: self
                                .operand(&fields[rustc_abi::FieldIdx::from_u32(1)], span)?,
                        })
                    }
                    Some(_) => self.pointer_cast(data, destination, span),
                    None => Ok(ir::Expression::Use(self.operand(data, span)?)),
                }
            }
            mir::Rvalue::Aggregate(_, fields) => {
                let ty = destination;
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
                    // A field-less value such as `RangeFull`.
                    ir::Type::Unit => Ok(ir::Expression::Use(ir::Operand::Constant(
                        ir::Constant::Unit,
                    ))),
                    ir::Type::Dim3 => Ok(ir::Expression::Dim3(
                        fields
                            .try_into()
                            .map_err(|_| (span, "invalid Dim3 aggregate".into()))?,
                    )),
                    _ => Err((span, "unsupported MIR aggregate type".into())),
                }
            }
            mir::Rvalue::Repeat(value, _) => {
                let ir::Type::Aggregate(ty) = destination else {
                    unreachable!()
                };
                let count = self.module.types.table().get(ty).unwrap().len();
                Ok(ir::Expression::Aggregate {
                    ty,
                    fields: vec![self.operand(value, span)?; count],
                })
            }
            mir::Rvalue::Use(v, _) => Ok(ir::Expression::Use(self.operand(v, span)?)),
            mir::Rvalue::CopyForDeref(place) => Ok(ir::Expression::Use(ir::Operand::Place(
                self.place(place, span)?,
            ))),
            mir::Rvalue::Ref(_, kind, place) => {
                let mutable = match kind {
                    mir::BorrowKind::Shared => false,
                    mir::BorrowKind::Mut { .. } => true,
                    mir::BorrowKind::Fake(_) => {
                        return Err((span, "fake borrows cannot be imported".into()));
                    }
                };
                Ok(ir::Expression::Ref {
                    place: ir::Operand::Place(self.place(place, span)?),
                    mutable,
                })
            }
            mir::Rvalue::RawPtr(kind, place) => Ok(ir::Expression::Ref {
                place: ir::Operand::Place(self.place(place, span)?),
                mutable: *kind == mir::RawPtrKind::Mut,
            }),
            mir::Rvalue::Cast(
                mir::CastKind::PtrToPtr
                | mir::CastKind::Transmute
                | mir::CastKind::PointerCoercion(PointerCoercion::MutToConstPointer, _),
                value,
                _,
            ) if self.module.types.table().pointer(destination).is_some() => {
                self.pointer_cast(value, destination, span)
            }
            mir::Rvalue::UnaryOp(mir::UnOp::PtrMetadata, v) => {
                Ok(ir::Expression::Len(self.operand(v, span)?))
            }
            mir::Rvalue::Cast(mir::CastKind::PointerCoercion(PointerCoercion::Unsize, _), v, _) => {
                Ok(ir::Expression::Slice(self.operand(v, span)?))
            }
            mir::Rvalue::Discriminant(place) => {
                let (mut place, ty) = self.typed_place(place, span)?;
                if ty == ir::Type::Scalar(ir::Scalar::I8) {
                    return Ok(ir::Expression::Use(ir::Operand::Place(place)));
                }
                if self.module.types.tag(ty).is_none() {
                    return Err((span, "discriminants require an enum".into()));
                }
                place.projection.push(ir::Projection::Field(0));
                Ok(ir::Expression::Use(ir::Operand::Place(place)))
            }
            mir::Rvalue::BinaryOp(op, values) => Ok(ir::Expression::Binary(
                binary(*op),
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
                if to.is_integer()
                    && let Some(value) =
                        super::constants::integer_cast(self.module.tcx, self.instance, v, to)
                {
                    return Ok(ir::Expression::Use(ir::Operand::Constant(value)));
                }
                Ok(ir::Expression::Cast(self.operand(v, span)?, to))
            }
            mir::Rvalue::Cast(mir::CastKind::Transmute, value, _) => {
                let from = self.operand_type(value, span)?;
                match (from, destination) {
                    // A restricted value, such as a `NonZero`, and the value it holds.
                    (from, to) if from == to => Ok(ir::Expression::Use(self.operand(value, span)?)),
                    (ir::Type::Scalar(from), ir::Type::Scalar(to))
                        if from.bits() == to.bits()
                            && from != ir::Scalar::Bool
                            && to != ir::Scalar::Bool =>
                    {
                        Ok(ir::Expression::Bitcast(self.operand(value, span)?, to))
                    }
                    _ => Err((
                        span,
                        "transmutes are supported between same-width numbers and between pointers"
                            .into(),
                    )),
                }
            }
            other => Err((span, format!("unsupported MIR rvalue: {other:?}"))),
        }
    }
}

/// MIR's operation; an unchecked one, whose overflow is undefined behavior, wraps.
fn binary(op: mir::BinOp) -> ir::BinaryOp {
    match op {
        mir::BinOp::Add | mir::BinOp::AddUnchecked => ir::BinaryOp::Add,
        mir::BinOp::Sub | mir::BinOp::SubUnchecked => ir::BinaryOp::Sub,
        mir::BinOp::Mul | mir::BinOp::MulUnchecked => ir::BinaryOp::Mul,
        mir::BinOp::Div => ir::BinaryOp::Div,
        mir::BinOp::Rem => ir::BinaryOp::Rem,
        mir::BinOp::BitAnd => ir::BinaryOp::BitAnd,
        mir::BinOp::BitOr => ir::BinaryOp::BitOr,
        mir::BinOp::BitXor => ir::BinaryOp::BitXor,
        mir::BinOp::Shl | mir::BinOp::ShlUnchecked => ir::BinaryOp::Shl,
        mir::BinOp::Shr | mir::BinOp::ShrUnchecked => ir::BinaryOp::Shr,
        mir::BinOp::Eq => ir::BinaryOp::Eq,
        mir::BinOp::Ne => ir::BinaryOp::Ne,
        mir::BinOp::Lt => ir::BinaryOp::Lt,
        mir::BinOp::Le => ir::BinaryOp::Le,
        mir::BinOp::Gt => ir::BinaryOp::Gt,
        mir::BinOp::Ge => ir::BinaryOp::Ge,
        mir::BinOp::Cmp => ir::BinaryOp::Cmp,
        mir::BinOp::AddWithOverflow => ir::BinaryOp::AddWithOverflow,
        mir::BinOp::SubWithOverflow => ir::BinaryOp::SubWithOverflow,
        mir::BinOp::MulWithOverflow => ir::BinaryOp::MulWithOverflow,
        mir::BinOp::Offset => ir::BinaryOp::Offset,
    }
}
