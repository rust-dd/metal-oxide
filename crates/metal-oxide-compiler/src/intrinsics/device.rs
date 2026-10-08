use metal_oxide_ir::{Access, AtomicOp, Builtin, MathOp, Scalar, SimdBuiltin};
use rustc_hir::{
    def::DefKind,
    def_id::{CRATE_DEF_INDEX, DefId},
};
use rustc_middle::ty::{self, TyCtxt};

use super::Intrinsic;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceType {
    ReadBuffer,
    WriteBuffer,
    AtomicBuffer,
    AtomicThreadgroupBuffer,
    ThreadgroupBuffer,
    Dim3,
    F16,
}

pub(crate) fn device_type(tcx: TyCtxt<'_>, definition: DefId) -> Option<DeviceType> {
    if tcx.def_kind(definition) != DefKind::Struct {
        return None;
    }
    let (module, kind) = match tcx.item_name(definition).as_str() {
        "ReadBuffer" => ("buffer", DeviceType::ReadBuffer),
        "WriteBuffer" => ("buffer", DeviceType::WriteBuffer),
        "AtomicThreadgroupBuffer" => ("atomic", DeviceType::AtomicThreadgroupBuffer),
        "AtomicBuffer" => ("atomic", DeviceType::AtomicBuffer),
        "ThreadgroupBuffer" => ("threadgroup", DeviceType::ThreadgroupBuffer),
        "Dim3" => ("thread", DeviceType::Dim3),
        "F16" => ("half", DeviceType::F16),
        _ => return None,
    };
    in_module(tcx, definition, module).then_some(kind)
}

pub(super) fn operation(tcx: TyCtxt<'_>, definition: DefId) -> Option<Intrinsic> {
    if tcx.is_foreign_item(definition) && in_module(tcx, definition, "intrinsics") {
        return match tcx.item_name(definition).as_str() {
            "__metal_thread_idx" => Some(Intrinsic::Coordinates(Builtin::ThreadIdx)),
            "__metal_block_idx" => Some(Intrinsic::Coordinates(Builtin::BlockIdx)),
            "__metal_block_dim" => Some(Intrinsic::Coordinates(Builtin::BlockDim)),
            "__metal_grid_dim" => Some(Intrinsic::Coordinates(Builtin::GridDim)),
            "__metal_sqrt_f32" => Some(Intrinsic::Math(MathOp::Sqrt)),
            "__metal_fma_f32" => Some(Intrinsic::Math(MathOp::Fma)),
            "__metal_f16_from_f32" => Some(Intrinsic::FloatConvert(Scalar::F16)),
            "__metal_f16_to_f32" => Some(Intrinsic::FloatConvert(Scalar::F32)),
            "__metal_threadgroup_barrier" => Some(Intrinsic::ThreadgroupBarrier),
            "__metal_simd_lane" => Some(Intrinsic::SimdCoordinate(SimdBuiltin::Lane)),
            "__metal_simd_size" => Some(Intrinsic::SimdCoordinate(SimdBuiltin::Size)),
            "__metal_simd_group" => Some(Intrinsic::SimdCoordinate(SimdBuiltin::Group)),
            "__metal_simd_count" => Some(Intrinsic::SimdCoordinate(SimdBuiltin::Count)),
            "__metal_simd_sum" => Some(Intrinsic::SimdSum),
            "__metal_simd_shuffle" => Some(Intrinsic::SimdShuffle),
            _ => None,
        };
    }
    if tcx.def_kind(definition) == DefKind::Fn
        && in_module(tcx, definition, "threadgroup")
        && matches!(
            tcx.item_name(definition).as_str(),
            "shared" | "shared_atomic"
        )
    {
        return Some(Intrinsic::ThreadgroupAlloc(
            if tcx.item_name(definition).as_str() == "shared_atomic" {
                Access::Atomic
            } else {
                Access::ReadWrite
            },
        ));
    }
    let implementation = tcx.inherent_impl_of_assoc(definition)?;
    let ty = tcx.normalize_erasing_regions(
        ty::TypingEnv::non_body_analysis(tcx, implementation),
        tcx.type_of(implementation).instantiate_identity(),
    );
    let ty::Adt(owner, _) = ty.kind() else {
        return None;
    };
    match (
        device_type(tcx, owner.did())?,
        tcx.item_name(definition).as_str(),
    ) {
        (DeviceType::ReadBuffer | DeviceType::ThreadgroupBuffer, "load_unchecked") => {
            Some(Intrinsic::BufferLoad)
        }
        (DeviceType::WriteBuffer | DeviceType::ThreadgroupBuffer, "store_unchecked") => {
            Some(Intrinsic::BufferStore)
        }
        (DeviceType::AtomicBuffer | DeviceType::AtomicThreadgroupBuffer, method) => {
            let op = match method {
                "load_relaxed" => AtomicOp::Load,
                "store_relaxed" => AtomicOp::Store,
                "exchange_relaxed" => AtomicOp::Exchange,
                "compare_exchange_weak_relaxed" => AtomicOp::CompareExchangeWeak,
                "fetch_add_relaxed" => AtomicOp::Add,
                "fetch_sub_relaxed" => AtomicOp::Sub,
                "fetch_min_relaxed" => AtomicOp::Min,
                "fetch_max_relaxed" => AtomicOp::Max,
                "fetch_and_relaxed" => AtomicOp::And,
                "fetch_or_relaxed" => AtomicOp::Or,
                "fetch_xor_relaxed" => AtomicOp::Xor,
                _ => return None,
            };
            Some(Intrinsic::Atomic(op))
        }
        _ => None,
    }
}

pub(super) fn in_module(tcx: TyCtxt<'_>, definition: DefId, module: &str) -> bool {
    if tcx.crate_name(definition.krate).as_str() != "metal_oxide_device" {
        return false;
    }
    let mut parent = tcx.parent(definition);
    if tcx.def_kind(parent) == DefKind::ForeignMod {
        parent = tcx.parent(parent);
    }
    tcx.def_kind(parent) == DefKind::Mod
        && tcx.item_name(parent).as_str() == module
        && tcx.parent(parent).index == CRATE_DEF_INDEX
}
