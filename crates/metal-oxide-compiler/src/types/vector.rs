use rustc_middle::ty::{self, Ty, TyCtxt, TypingEnv, consts::ConstExt};
use rustc_span::Span;

pub(super) fn validate<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    span: Span,
) -> Result<(), (Span, String)> {
    let ty::Adt(adt, args) = ty.kind() else {
        return Ok(());
    };
    if !adt.repr().simd() {
        return Ok(());
    }
    let valid = adt.non_enum_variant().fields.len() == 1
        && {
            let field = tcx.normalize_erasing_regions(
                TypingEnv::fully_monomorphized(),
                adt.non_enum_variant().fields[rustc_abi::FieldIdx::ZERO].ty(tcx, args),
            );
            matches!(field.kind(), ty::Array(element, count)
            if matches!(super::scalar(*element), Some(metal_oxide_ir::Scalar::F32 | metal_oxide_ir::Scalar::I32 | metal_oxide_ir::Scalar::U32 | metal_oxide_ir::Scalar::I8 | metal_oxide_ir::Scalar::U8 | metal_oxide_ir::Scalar::I16 | metal_oxide_ir::Scalar::U16))
            && matches!(count.try_to_target_usize(tcx), Some(2..=4)))
        };
    if valid {
        Ok(())
    } else {
        Err((
            span,
            "vectors require 2-4 lanes of f32 or 8/16/32-bit integers".into(),
        ))
    }
}
