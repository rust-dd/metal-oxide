use metal_oxide_ir::{BinaryOp, Builtin, SimdBuiltin};
use rustc_hir::def_id::DefId;
use rustc_middle::ty::TyCtxt;
use rustc_span::Symbol;

#[derive(Clone, Copy)]
pub(crate) enum Intrinsic {
    Coordinates(Builtin),
    SimdCoordinate(SimdBuiltin),
    BufferLoad,
    BufferStore,
    ThreadgroupAlloc,
    ThreadgroupBarrier,
    SimdSum,
    SimdShuffle,
    AtomicAdd,
    WrappingShift(BinaryOp),
}

impl Intrinsic {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Coordinates(Builtin::ThreadIdx) => "thread_idx",
            Self::Coordinates(Builtin::BlockIdx) => "block_idx",
            Self::Coordinates(Builtin::BlockDim) => "block_dim",
            Self::Coordinates(Builtin::GridDim) => "grid_dim",
            Self::SimdCoordinate(SimdBuiltin::Lane) => "simd_lane",
            Self::SimdCoordinate(SimdBuiltin::Size) => "simd_size",
            Self::SimdCoordinate(SimdBuiltin::Group) => "simd_group",
            Self::SimdCoordinate(SimdBuiltin::Count) => "simd_count",
            Self::BufferLoad => "buffer_load",
            Self::BufferStore => "buffer_store",
            Self::ThreadgroupAlloc => "threadgroup_alloc",
            Self::ThreadgroupBarrier => "threadgroup_barrier",
            Self::SimdSum => "simd_sum",
            Self::SimdShuffle => "simd_shuffle",
            Self::AtomicAdd => "atomic_add",
            Self::WrappingShift(_) => "wrapping_shift",
        }
    }
}

pub(crate) fn builtin(tcx: TyCtxt<'_>, definition: DefId) -> Option<Intrinsic> {
    if tcx.crate_name(definition.krate).as_str() == "core"
        && tcx.def_path_str(definition).starts_with("core::num::")
    {
        match tcx.item_name(definition).as_str() {
            "wrapping_shl" => return Some(Intrinsic::WrappingShift(BinaryOp::Shl)),
            "wrapping_shr" => return Some(Intrinsic::WrappingShift(BinaryOp::Shr)),
            _ => {}
        }
    }
    [
        ("metal_oxide_buffer_load", Intrinsic::BufferLoad),
        ("metal_oxide_buffer_store", Intrinsic::BufferStore),
        ("metal_oxide_threadgroup_alloc", Intrinsic::ThreadgroupAlloc),
        ("metal_oxide_threadgroup_load", Intrinsic::BufferLoad),
        ("metal_oxide_threadgroup_store", Intrinsic::BufferStore),
        (
            "metal_oxide_threadgroup_barrier",
            Intrinsic::ThreadgroupBarrier,
        ),
        (
            "metal_oxide_simd_lane",
            Intrinsic::SimdCoordinate(SimdBuiltin::Lane),
        ),
        (
            "metal_oxide_simd_size",
            Intrinsic::SimdCoordinate(SimdBuiltin::Size),
        ),
        (
            "metal_oxide_simd_group",
            Intrinsic::SimdCoordinate(SimdBuiltin::Group),
        ),
        (
            "metal_oxide_simd_count",
            Intrinsic::SimdCoordinate(SimdBuiltin::Count),
        ),
        ("metal_oxide_simd_sum", Intrinsic::SimdSum),
        ("metal_oxide_simd_shuffle", Intrinsic::SimdShuffle),
        ("metal_oxide_atomic_add", Intrinsic::AtomicAdd),
        (
            "metal_oxide_thread_idx",
            Intrinsic::Coordinates(Builtin::ThreadIdx),
        ),
        (
            "metal_oxide_block_idx",
            Intrinsic::Coordinates(Builtin::BlockIdx),
        ),
        (
            "metal_oxide_block_dim",
            Intrinsic::Coordinates(Builtin::BlockDim),
        ),
        (
            "metal_oxide_grid_dim",
            Intrinsic::Coordinates(Builtin::GridDim),
        ),
    ]
    .into_iter()
    .find_map(|(item, builtin)| {
        (tcx.get_diagnostic_item(Symbol::intern(item)) == Some(definition)).then_some(builtin)
    })
}
