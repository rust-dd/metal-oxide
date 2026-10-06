use rustc_hir::def_id::DefId;
use rustc_middle::ty::TyCtxt;
use rustc_span::Symbol;

pub(crate) fn builtin(tcx: TyCtxt<'_>, definition: DefId) -> Option<&'static str> {
    [
        ("metal_oxide_buffer_load", "buffer_load"),
        ("metal_oxide_buffer_store", "buffer_store"),
        ("metal_oxide_threadgroup_alloc", "threadgroup_alloc"),
        ("metal_oxide_threadgroup_load", "buffer_load"),
        ("metal_oxide_threadgroup_store", "buffer_store"),
        ("metal_oxide_threadgroup_barrier", "threadgroup_barrier"),
        ("metal_oxide_simd_lane", "simd_lane"),
        ("metal_oxide_simd_size", "simd_size"),
        ("metal_oxide_simd_group", "simd_group"),
        ("metal_oxide_simd_count", "simd_count"),
        ("metal_oxide_simd_sum", "simd_sum"),
        ("metal_oxide_simd_shuffle", "simd_shuffle"),
        ("metal_oxide_atomic_add", "atomic_add"),
        ("metal_oxide_thread_idx", "thread_idx"),
        ("metal_oxide_block_idx", "block_idx"),
        ("metal_oxide_block_dim", "block_dim"),
        ("metal_oxide_grid_dim", "grid_dim"),
    ]
    .into_iter()
    .find_map(|(item, builtin)| {
        (tcx.get_diagnostic_item(Symbol::intern(item)) == Some(definition)).then_some(builtin)
    })
}
