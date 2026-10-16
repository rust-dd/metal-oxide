use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::Symbol;

#[derive(Clone, Copy)]
pub(crate) enum VectorOp {
    Intrinsic(Symbol),
    FromArray,
    ToArray,
    AsArray(bool),
    MaskFromArray,
    MaskToArray,
    ToBitmask,
    FromBitmask,
}

pub(super) fn operation(tcx: TyCtxt<'_>, definition: DefId) -> Option<VectorOp> {
    let implementation = tcx.inherent_impl_of_assoc(definition)?;
    let ty = tcx.normalize_erasing_regions(
        ty::TypingEnv::non_body_analysis(tcx, implementation),
        tcx.type_of(implementation).instantiate_identity(),
    );
    let ty::Adt(adt, _) = ty.kind() else {
        return None;
    };
    if adt.repr().simd() {
        return match tcx.item_name(definition).as_str() {
            "from_array" => Some(VectorOp::FromArray),
            "to_array" => Some(VectorOp::ToArray),
            "as_array" => Some(VectorOp::AsArray(false)),
            "as_mut_array" => Some(VectorOp::AsArray(true)),
            _ => None,
        };
    }
    if tcx.def_path_str(adt.did()) == "core::simd::Mask" {
        return match tcx.item_name(definition).as_str() {
            "from_array" => Some(VectorOp::MaskFromArray),
            "to_array" => Some(VectorOp::MaskToArray),
            "to_bitmask" => Some(VectorOp::ToBitmask),
            "from_bitmask" => Some(VectorOp::FromBitmask),
            _ => None,
        };
    }
    None
}
