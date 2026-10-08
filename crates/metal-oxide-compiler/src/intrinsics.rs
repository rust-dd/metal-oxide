use metal_oxide_ir::{Access, AtomicOp, BinaryOp, Builtin, MathOp, Scalar, SimdBuiltin, SimdOp};
mod device;
mod signatures;

use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, TyCtxt};

pub(crate) use device::{DeviceType, device_type};

#[derive(Clone, Copy)]
pub(crate) enum Intrinsic {
    Coordinates(Builtin),
    SimdCoordinate(SimdBuiltin),
    BufferLoad,
    BufferStore,
    ThreadgroupAlloc(Access),
    ThreadgroupBarrier,
    Simd(SimdOp),
    Atomic(AtomicOp),
    WrappingShift(BinaryOp),
    Math(MathOp),
    Bitcast(Scalar),
    FloatConvert(Scalar),
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
            Self::ThreadgroupAlloc(_) => "threadgroup_alloc",
            Self::ThreadgroupBarrier => "threadgroup_barrier",
            Self::Simd(SimdOp::Sum) => "simd_sum",
            Self::Simd(SimdOp::Shuffle) => "simd_shuffle",
            Self::Simd(_) => "simd",
            Self::Atomic(AtomicOp::Add) => "atomic_add",
            Self::Atomic(_) => "atomic",
            Self::WrappingShift(_) => "wrapping_shift",
            Self::Math(_) => "float_math",
            Self::Bitcast(_) => "scalar_bitcast",
            Self::FloatConvert(_) => "float_conversion",
        }
    }
}

pub(crate) fn builtin(tcx: TyCtxt<'_>, definition: DefId) -> Option<Intrinsic> {
    if let Some(operation) = device::operation(tcx, definition) {
        if !signatures::valid(tcx, definition, operation) {
            tcx.dcx().span_fatal(
                tcx.def_span(definition),
                "invalid device intrinsic signature",
            );
        }
        return Some(operation);
    }
    if tcx.crate_name(definition.krate).as_str() != "core" {
        return None;
    }
    let implementation = tcx.inherent_impl_of_assoc(definition)?;
    let ty = tcx.normalize_erasing_regions(
        ty::TypingEnv::non_body_analysis(tcx, implementation),
        tcx.type_of(implementation).instantiate_identity(),
    );
    if matches!(ty.kind(), ty::Float(ty::FloatTy::F32)) {
        match tcx.item_name(definition).as_str() {
            "abs" => return Some(Intrinsic::Math(MathOp::Abs)),
            "min" => return Some(Intrinsic::Math(MathOp::Min)),
            "max" => return Some(Intrinsic::Math(MathOp::Max)),
            "to_bits" => return Some(Intrinsic::Bitcast(Scalar::U32)),
            "from_bits" => return Some(Intrinsic::Bitcast(Scalar::F32)),
            _ => {}
        }
    }
    if matches!(ty.kind(), ty::Int(_) | ty::Uint(_)) {
        match tcx.item_name(definition).as_str() {
            "wrapping_shl" => return Some(Intrinsic::WrappingShift(BinaryOp::Shl)),
            "wrapping_shr" => return Some(Intrinsic::WrappingShift(BinaryOp::Shr)),
            _ => {}
        }
    }
    None
}
