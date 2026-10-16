use rustc_middle::ty::{self, Ty, TyCtxt, TypingEnv};

pub(super) fn contains_internal_type<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    if super::ordering(tcx, ty) {
        return true;
    }
    if let Some((_, _, element)) = super::buffer(tcx, ty) {
        return contains_internal_type(tcx, element);
    }
    match ty.kind() {
        ty::Adt(definition, args) => {
            definition.repr().simd()
                || definition.all_fields().any(|f| {
                    contains_internal_type(
                        tcx,
                        tcx.normalize_erasing_regions(
                            TypingEnv::fully_monomorphized(),
                            f.ty(tcx, args),
                        ),
                    )
                })
        }
        ty::Array(element, _) | ty::Slice(element) => contains_internal_type(tcx, *element),
        ty::Tuple(fields) => fields.iter().any(|ty| contains_internal_type(tcx, ty)),
        // References and pointers cannot cross the ABI, so do not follow recursive pointees.
        _ => false,
    }
}
