use metal_oxide_ir::{Access, AtomicOp, BinaryOp, Builtin, MathOp, Scalar, SimdBuiltin, SimdOp};
use rustc_hir::{
    def::DefKind,
    def_id::{CRATE_DEF_INDEX, DefId},
};
use rustc_middle::ty::{self, TyCtxt};

use super::{BufferQuery, Intrinsic};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceType {
    ReadBuffer,
    WriteBuffer,
    DisjointBuffer,
    ReadWriteBuffer,
    AtomicBuffer,
    AtomicThreadgroupBuffer,
    ThreadgroupBuffer,
    DynamicThreadgroupBuffer,
    Dim3,
    F16,
    BF16,
    ThreadIndex,
}

pub(crate) fn device_type(tcx: TyCtxt<'_>, definition: DefId) -> Option<DeviceType> {
    if tcx.def_kind(definition) != DefKind::Struct {
        return None;
    }
    let (module, kind) = match tcx.item_name(definition).as_str() {
        "ReadBuffer" => ("buffer", DeviceType::ReadBuffer),
        "WriteBuffer" => ("buffer", DeviceType::WriteBuffer),
        "DisjointBuffer" => ("buffer", DeviceType::DisjointBuffer),
        "ReadWriteBuffer" => ("buffer", DeviceType::ReadWriteBuffer),
        "AtomicThreadgroupBuffer" => ("atomic", DeviceType::AtomicThreadgroupBuffer),
        "AtomicBuffer" => ("atomic", DeviceType::AtomicBuffer),
        "ThreadgroupBuffer" => ("threadgroup", DeviceType::ThreadgroupBuffer),
        "DynamicThreadgroupBuffer" => ("threadgroup", DeviceType::DynamicThreadgroupBuffer),
        "Dim3" => ("thread", DeviceType::Dim3),
        "ThreadIndex" => ("thread", DeviceType::ThreadIndex),
        "F16" => ("half", DeviceType::F16),
        "BF16" => ("bfloat", DeviceType::BF16),
        _ => return None,
    };
    in_module(tcx, definition, module).then_some(kind)
}

pub(super) fn operation(tcx: TyCtxt<'_>, definition: DefId) -> Option<Intrinsic> {
    if tcx.is_foreign_item(definition) && in_module(tcx, definition, "intrinsics") {
        let name = tcx.item_name(definition);
        for (prefix, scalar) in [
            ("__metal_f16_", Scalar::F16),
            ("__metal_bf16_", Scalar::BF16),
        ] {
            if let Some(operation) = name.as_str().strip_prefix(prefix) {
                return Some(match operation {
                    "from_f32" => Intrinsic::FloatConvert {
                        from: Scalar::F32,
                        to: scalar,
                    },
                    "to_f32" => Intrinsic::FloatConvert {
                        from: scalar,
                        to: Scalar::F32,
                    },
                    "add" => Intrinsic::FloatBinary(scalar, BinaryOp::Add),
                    "sub" => Intrinsic::FloatBinary(scalar, BinaryOp::Sub),
                    "mul" => Intrinsic::FloatBinary(scalar, BinaryOp::Mul),
                    "div" => Intrinsic::FloatBinary(scalar, BinaryOp::Div),
                    "rem" => Intrinsic::FloatBinary(scalar, BinaryOp::Rem),
                    _ => return None,
                });
            }
        }
        return match name.as_str() {
            "__metal_thread_idx" => Some(Intrinsic::Coordinates(Builtin::ThreadIdx)),
            "__metal_block_idx" => Some(Intrinsic::Coordinates(Builtin::BlockIdx)),
            "__metal_block_dim" => Some(Intrinsic::Coordinates(Builtin::BlockDim)),
            "__metal_grid_dim" => Some(Intrinsic::Coordinates(Builtin::GridDim)),
            "__metal_thread_index" => Some(Intrinsic::ThreadIndex),
            "__metal_sqrt_f32" => Some(Intrinsic::Math(MathOp::Sqrt)),
            "__metal_fma_f32" => Some(Intrinsic::Math(MathOp::Fma)),
            "__metal_ln_f32" => Some(Intrinsic::Math(MathOp::Ln)),
            "__metal_exp_f32" => Some(Intrinsic::Math(MathOp::Exp)),
            "__metal_sin_f32" => Some(Intrinsic::Math(MathOp::Sin)),
            "__metal_cos_f32" => Some(Intrinsic::Math(MathOp::Cos)),
            "__metal_sin_cos_f32" => Some(Intrinsic::Math(MathOp::SinCos)),
            "__metal_powf_f32" => Some(Intrinsic::Math(MathOp::Powf)),
            "__metal_tanh_f32" => Some(Intrinsic::Math(MathOp::Tanh)),
            "__metal_atan_f32" => Some(Intrinsic::Math(MathOp::Atan)),
            "__metal_floor_f32" => Some(Intrinsic::Math(MathOp::Floor)),
            "__metal_round_f32" => Some(Intrinsic::Math(MathOp::Round)),
            "__metal_ceil_f32" => Some(Intrinsic::Math(MathOp::Ceil)),
            "__metal_trunc_f32" => Some(Intrinsic::Math(MathOp::Trunc)),
            "__metal_threadgroup_barrier" => Some(Intrinsic::ThreadgroupBarrier),
            "__metal_simd_lane" => Some(Intrinsic::SimdCoordinate(SimdBuiltin::Lane)),
            "__metal_simd_size" => Some(Intrinsic::SimdCoordinate(SimdBuiltin::Size)),
            "__metal_simd_group" => Some(Intrinsic::SimdCoordinate(SimdBuiltin::Group)),
            "__metal_simd_count" => Some(Intrinsic::SimdCoordinate(SimdBuiltin::Count)),
            "__metal_simd_any" => Some(Intrinsic::Simd(SimdOp::Any)),
            "__metal_simd_all" => Some(Intrinsic::Simd(SimdOp::All)),
            "__metal_simd_ballot" => Some(Intrinsic::Simd(SimdOp::Ballot)),
            _ => None,
        };
    }
    if tcx.def_kind(definition) == DefKind::Fn && in_module(tcx, definition, "simdgroup") {
        let op = match tcx.item_name(definition).as_str() {
            "sum" => SimdOp::Sum,
            "min" => SimdOp::Min,
            "max" => SimdOp::Max,
            "and" => SimdOp::And,
            "or" => SimdOp::Or,
            "xor" => SimdOp::Xor,
            "inclusive_sum" => SimdOp::InclusiveSum,
            "exclusive_sum" => SimdOp::ExclusiveSum,
            "shuffle" => SimdOp::Shuffle,
            "shuffle_up" => SimdOp::ShuffleUp,
            "shuffle_down" => SimdOp::ShuffleDown,
            "shuffle_xor" => SimdOp::ShuffleXor,
            _ => return None,
        };
        return Some(Intrinsic::Simd(op));
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
    if tcx.def_kind(definition) == DefKind::Fn
        && in_module(tcx, definition, "thread")
        && tcx.item_name(definition).as_str() == "uninit_array"
    {
        return Some(Intrinsic::Uninit);
    }
    if tcx.def_kind(definition) == DefKind::Fn
        && in_module(tcx, definition, "log")
        && tcx.item_name(definition).as_str() == "write"
    {
        return Some(Intrinsic::Log);
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
        (
            DeviceType::ReadBuffer
            | DeviceType::ReadWriteBuffer
            | DeviceType::ThreadgroupBuffer
            | DeviceType::DynamicThreadgroupBuffer,
            "load_unchecked",
        ) => Some(Intrinsic::BufferLoad),
        (
            DeviceType::WriteBuffer
            | DeviceType::ReadWriteBuffer
            | DeviceType::ThreadgroupBuffer
            | DeviceType::DynamicThreadgroupBuffer,
            "store_unchecked",
        ) => Some(Intrinsic::BufferStore),
        (
            DeviceType::ReadBuffer
            | DeviceType::WriteBuffer
            | DeviceType::ReadWriteBuffer
            | DeviceType::DisjointBuffer,
            "len",
        ) => Some(Intrinsic::Buffer(BufferQuery::Length)),
        (
            DeviceType::ReadBuffer
            | DeviceType::WriteBuffer
            | DeviceType::ReadWriteBuffer
            | DeviceType::DisjointBuffer,
            "is_empty",
        ) => Some(Intrinsic::Buffer(BufferQuery::IsEmpty)),
        (DeviceType::ReadBuffer, "as_slice") => Some(Intrinsic::Buffer(BufferQuery::Slice)),
        (DeviceType::DisjointBuffer, "get_mut") => Some(Intrinsic::Buffer(BufferQuery::Element)),
        (DeviceType::ThreadIndex, "get") => Some(Intrinsic::ThreadIndex),
        (DeviceType::AtomicBuffer | DeviceType::AtomicThreadgroupBuffer, method) => {
            let op = match method {
                "load" => AtomicOp::Load,
                "store" => AtomicOp::Store,
                "exchange" => AtomicOp::Exchange,
                "compare_exchange_weak" => AtomicOp::CompareExchangeWeak,
                "fetch_add" => AtomicOp::Add,
                "fetch_sub" => AtomicOp::Sub,
                "fetch_min" => AtomicOp::Min,
                "fetch_max" => AtomicOp::Max,
                "fetch_and" => AtomicOp::And,
                "fetch_or" => AtomicOp::Or,
                "fetch_xor" => AtomicOp::Xor,
                "min" => AtomicOp::MinStore,
                "max" => AtomicOp::MaxStore,
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
