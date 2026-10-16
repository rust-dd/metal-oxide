use super::super::Result;
use rustc_middle::ty::{self, Ty, TyCtxt, TypingEnv, consts::ConstExt};
use rustc_span::Span;

pub(super) fn intrinsic<'tcx>(
    tcx: TyCtxt<'tcx>,
    name: &str,
    arguments: &[Ty<'tcx>],
    result: Ty<'tcx>,
    span: Span,
) -> Result<()> {
    let shape =
        |ty| vector(tcx, ty).ok_or((span, "SIMD intrinsics require repr(simd) vectors".into()));
    let invalid = || (span, format!("invalid SIMD intrinsic signature: {name}"));
    if name == "simd_splat" {
        return (shape(result)?.0 == arguments[0])
            .then_some(())
            .ok_or_else(invalid);
    }
    let (element, count) = shape(arguments[0])?;
    let valid = match name {
        "simd_extract" | "simd_extract_dyn" => result == element,
        "simd_insert" | "simd_insert_dyn" => result == arguments[0] && arguments[2] == element,
        "simd_bitmask" => {
            if !element.is_integral() || !matches!(result.kind(), ty::Uint(ty::UintTy::U8)) {
                return Err((
                    span,
                    "vector bitmask requires integer lanes and a u8 result".into(),
                ));
            }
            true
        }
        "simd_reduce_any" | "simd_reduce_all" => element.is_integral() && result.is_bool(),
        name if name.starts_with("simd_reduce_") => {
            result == element && arguments.get(1).is_none_or(|ty| *ty == element)
        }
        "simd_shuffle" => {
            let (output, length) = shape(result)?;
            let indices = array(tcx, arguments[2]).or_else(|| vector(tcx, arguments[2]));
            arguments[0] == arguments[1]
                && output == element
                && indices.is_some_and(|(ty, n)| ty == tcx.types.u32 && n == length)
        }
        "simd_select" => {
            let (_, length) = shape(arguments[1])?;
            element.is_integral()
                && length == count
                && arguments[1] == arguments[2]
                && result == arguments[1]
        }
        "simd_eq" | "simd_ne" | "simd_lt" | "simd_le" | "simd_gt" | "simd_ge" => {
            let (mask, length) = shape(result)?;
            arguments[0] == arguments[1]
                && length == count
                && mask.is_integral()
                && crate::types::scalar(element).map(|s| s.bits())
                    == crate::types::scalar(mask).map(|s| s.bits())
        }
        "simd_as" | "simd_cast" => shape(result)?.1 == count,
        _ => {
            shape(result)?;
            result == arguments[0] && arguments.iter().all(|ty| *ty == arguments[0])
        }
    };
    valid.then_some(()).ok_or_else(invalid)
}

fn vector<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<(Ty<'tcx>, u64)> {
    let ty::Adt(adt, args) = ty.kind() else {
        return None;
    };
    if !adt.repr().simd() || adt.non_enum_variant().fields.len() != 1 {
        return None;
    }
    let field = tcx.normalize_erasing_regions(
        TypingEnv::fully_monomorphized(),
        adt.non_enum_variant().fields[rustc_abi::FieldIdx::ZERO].ty(tcx, args),
    );
    array(tcx, field)
}

fn array<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<(Ty<'tcx>, u64)> {
    let ty::Array(element, count) = ty.kind() else {
        return None;
    };
    Some((*element, count.try_to_target_usize(tcx)?))
}
