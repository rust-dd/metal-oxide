use metal_oxide_ir as ir;
use rustc_abi::{Size, TagEncoding, VariantIdx, Variants};
use rustc_middle::{
    mir::{
        self,
        interpret::{Allocation, Scalar, alloc_range},
    },
    ty::{self, Ty, TyCtxt, TypingEnv, layout::LayoutCx},
};
use rustc_span::Span;

use super::{Result, function::FunctionImporter};

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(super) fn owned_literal(
        &mut self,
        constant: mir::Const<'tcx>,
        span: Span,
    ) -> Result<ir::Operand> {
        let tcx = self.module.tcx;
        let ty = constant.ty();
        let value = constant
            .eval(tcx, TypingEnv::fully_monomorphized(), span)
            .map_err(|_| (span, "cannot evaluate owned device constant".into()))?;
        match value {
            mir::ConstValue::Indirect { alloc_id, offset } => {
                let allocation = tcx.global_alloc(alloc_id).unwrap_memory();
                self.literal_at(ty, allocation.inner(), offset, span)
            }
            mir::ConstValue::Scalar(Scalar::Int(value)) => {
                let layout = tcx
                    .layout_of(TypingEnv::fully_monomorphized().as_query_input(ty))
                    .map_err(|e| (span, e.to_string()))?;
                let bytes = value.to_bits(layout.size).to_le_bytes();
                let allocation = Allocation::from_bytes_byte_aligned_immutable(
                    &bytes[..layout.size.bytes_usize()],
                    (),
                );
                self.literal_at(ty, &allocation, Size::ZERO, span)
            }
            _ => Err((span, "unsupported owned device constant".into())),
        }
    }

    fn literal_at(
        &mut self,
        ty: Ty<'tcx>,
        allocation: &Allocation,
        offset: Size,
        span: Span,
    ) -> Result<ir::Operand> {
        let tcx = self.module.tcx;
        let layout = tcx
            .layout_of(TypingEnv::fully_monomorphized().as_query_input(ty))
            .map_err(|e| (span, e.to_string()))?;
        let lowered = self.lower_type(ty, span)?;
        if lowered == ir::Type::Unit {
            return Ok(ir::Operand::Constant(ir::Constant::Unit));
        }
        if let ir::Type::Scalar(scalar) = lowered {
            let value = bits(tcx, allocation, offset, layout.size, span)?;
            return Ok(ir::Operand::Constant(constant(scalar, value as u64)));
        }
        if let ir::Type::Aggregate(id) = lowered
            && let Some(aggregate @ ir::Aggregate::Enum { .. }) =
                self.module.types.table().get(id).cloned()
        {
            return self.variant_literal(ty, lowered, &aggregate, allocation, offset, span);
        }
        let count = self.module.types.table().field_count(lowered);
        let fields = (0..count)
            .map(|index| {
                self.literal_at(
                    layout
                        .field(
                            &rustc_middle::ty::layout::LayoutCx::new(
                                tcx,
                                TypingEnv::fully_monomorphized(),
                            ),
                            index,
                        )
                        .ty,
                    allocation,
                    offset + layout.fields.offset(index),
                    span,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ir::Operand::AggregateConstant {
            ty: lowered,
            fields,
        })
    }

    /// An enum constant: the tag of its variant, the variant's fields in their slots and zeros
    /// in the other slots.
    fn variant_literal(
        &mut self,
        ty: Ty<'tcx>,
        lowered: ir::Type,
        aggregate: &ir::Aggregate,
        allocation: &Allocation,
        offset: Size,
        span: Span,
    ) -> Result<ir::Operand> {
        let tcx = self.module.tcx;
        let cx = LayoutCx::new(tcx, TypingEnv::fully_monomorphized());
        let layout = tcx
            .layout_of(TypingEnv::fully_monomorphized().as_query_input(ty))
            .map_err(|e| (span, e.to_string()))?;
        let variant = match &layout.variants {
            Variants::Single { index } => *index,
            Variants::Multiple {
                tag,
                tag_encoding,
                tag_field,
                ..
            } => {
                let size = tag.size(&tcx);
                let value = bits(
                    tcx,
                    allocation,
                    offset + layout.fields.offset(tag_field.as_usize()),
                    size,
                    span,
                )?;
                match tag_encoding {
                    TagEncoding::Direct => {
                        let ty::Adt(definition, _) = ty.kind() else {
                            return Err((span, "unsupported enum constant".into()));
                        };
                        definition
                            .discriminants(tcx)
                            .find(|(_, discriminant)| size.truncate(discriminant.val) == value)
                            .map(|(variant, _)| variant)
                            .ok_or_else(|| (span, "enum constant has no variant".to_string()))?
                    }
                    TagEncoding::Niche {
                        untagged_variant,
                        niche_variants,
                        niche_start,
                    } => {
                        let relative = size.truncate(value.wrapping_sub(*niche_start));
                        let last = niche_variants.last.as_u32() - niche_variants.start.as_u32();
                        if relative <= u128::from(last) {
                            VariantIdx::from_u32(niche_variants.start.as_u32() + relative as u32)
                        } else {
                            *untagged_variant
                        }
                    }
                }
            }
            _ => return Err((span, "enum constant of an uninhabited type".into())),
        };
        let table = self.module.types.table();
        let mut fields = (0..aggregate.len() as u32)
            .map(|index| zero(table, aggregate.field(index)?))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| (span, "unsupported enum constant payload".to_string()))?;
        let tag = aggregate.tag().unwrap();
        fields[0] = ir::Operand::Constant(constant(tag, u64::from(variant.as_u32())));
        let fields_layout = layout.for_variant(&cx, variant);
        for index in 0..fields_layout.fields.count() {
            let slot = aggregate.payload(variant.as_u32(), index as u32).unwrap();
            fields[slot as usize] = self.literal_at(
                fields_layout.field(&cx, index).ty,
                allocation,
                offset + fields_layout.fields.offset(index),
                span,
            )?;
        }
        Ok(ir::Operand::AggregateConstant {
            ty: lowered,
            fields,
        })
    }
}

/// The little-endian integer in `size` bytes at `offset`.
fn bits(
    tcx: TyCtxt<'_>,
    allocation: &Allocation,
    offset: Size,
    size: Size,
    span: Span,
) -> Result<u128> {
    let bytes = allocation
        .get_bytes_strip_provenance(&tcx, alloc_range(offset, size))
        .map_err(|_| {
            (
                span,
                "device constant contains uninitialized or pointer bytes".to_string(),
            )
        })?;
    let mut bits = [0_u8; 16];
    bits[..bytes.len()].copy_from_slice(bytes);
    Ok(u128::from_le_bytes(bits))
}

fn constant(scalar: ir::Scalar, value: u64) -> ir::Constant {
    match scalar {
        ir::Scalar::Bool => ir::Constant::Bool(value != 0),
        ir::Scalar::F32 => ir::Constant::F32(value as u32),
        ir::Scalar::F16 => ir::Constant::F16(value as u16),
        ir::Scalar::BF16 => ir::Constant::BF16(value as u16),
        ir::Scalar::U32 => ir::Constant::U32(value as u32),
        ir::Scalar::Usize => ir::Constant::Usize(value),
        ir::Scalar::U64 => ir::Constant::U64(value),
        ir::Scalar::I32 => ir::Constant::I32(value as i32),
        ir::Scalar::U8 => ir::Constant::U8(value as u8),
        ir::Scalar::U16 => ir::Constant::U16(value as u16),
        ir::Scalar::I8 => ir::Constant::I8(value as i8),
        ir::Scalar::I16 => ir::Constant::I16(value as i16),
        ir::Scalar::Isize => ir::Constant::Isize(value as i64),
    }
}

/// The zero value of `ty`, which fills the slots of an enum constant that its variant leaves.
fn zero(table: &ir::TypeTable, ty: ir::Type) -> Option<ir::Operand> {
    Some(match ty {
        ir::Type::Unit => ir::Operand::Constant(ir::Constant::Unit),
        ir::Type::Scalar(scalar) => ir::Operand::Constant(constant(scalar, 0)),
        ir::Type::Aggregate(id) => {
            let aggregate = table.get(id)?;
            ir::Operand::AggregateConstant {
                ty,
                fields: (0..aggregate.len() as u32)
                    .map(|index| zero(table, aggregate.field(index)?))
                    .collect::<Option<_>>()?,
            }
        }
        _ => return None,
    })
}
