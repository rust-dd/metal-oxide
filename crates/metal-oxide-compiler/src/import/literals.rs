use metal_oxide_ir as ir;
use rustc_abi::Size;
use rustc_middle::{
    mir::{
        self,
        interpret::{Allocation, Scalar, alloc_range},
    },
    ty::{Ty, TypingEnv},
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
        if let ir::Type::Scalar(scalar) = lowered {
            let bytes = allocation
                .get_bytes_strip_provenance(&tcx, alloc_range(offset, layout.size))
                .map_err(|_| {
                    (
                        span,
                        "device constant contains uninitialized or pointer bytes".into(),
                    )
                })?;
            let mut bits = [0_u8; 8];
            bits[..bytes.len()].copy_from_slice(bytes);
            let value = u64::from_le_bytes(bits);
            return Ok(ir::Operand::Constant(match scalar {
                ir::Scalar::Bool => ir::Constant::Bool(value != 0),
                ir::Scalar::F32 => ir::Constant::F32(value as u32),
                ir::Scalar::U32 => ir::Constant::U32(value as u32),
                ir::Scalar::Usize => ir::Constant::Usize(value),
                ir::Scalar::I32 => ir::Constant::I32(value as i32),
                ir::Scalar::U8 => ir::Constant::U8(value as u8),
                ir::Scalar::U16 => ir::Constant::U16(value as u16),
                ir::Scalar::I8 => ir::Constant::I8(value as i8),
                ir::Scalar::I16 => ir::Constant::I16(value as i16),
            }));
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
}
