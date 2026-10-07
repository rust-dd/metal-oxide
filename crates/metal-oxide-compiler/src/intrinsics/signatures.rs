use rustc_hir::{Mutability, Safety, def_id::DefId};
use rustc_middle::{
    mir,
    ty::{self, Ty, TyCtxt},
};

use super::{
    Intrinsic,
    device::{self, DeviceType},
};

#[derive(Clone, Copy)]
enum Slot {
    F32,
    U32,
    Usize,
    Dim3,
    F16,
    Pointer(Mutability),
    Unit,
}

pub(super) fn valid(tcx: TyCtxt<'_>, definition: DefId, operation: Intrinsic) -> bool {
    use Slot::*;
    if !tcx.is_foreign_item(definition) {
        return adapter(tcx, definition, operation);
    }
    match operation {
        Intrinsic::Coordinates(_) => declaration(tcx, definition, &[], Dim3, Safety::Safe),
        Intrinsic::SimdCoordinate(_) => declaration(tcx, definition, &[], U32, Safety::Safe),
        Intrinsic::Math(op) => {
            let signature = tcx.fn_sig(definition).instantiate_identity().skip_binder();
            foreign_header(tcx, definition, Safety::Safe)
                && signature.inputs().len() == op.arity()
                && signature.inputs().iter().all(|&ty| ty == tcx.types.f32)
                && signature.output() == tcx.types.f32
        }
        Intrinsic::FloatConvert(metal_oxide_ir::Scalar::F16) => {
            declaration(tcx, definition, &[F32], F16, Safety::Safe)
        }
        Intrinsic::FloatConvert(metal_oxide_ir::Scalar::F32) => {
            declaration(tcx, definition, &[F16], F32, Safety::Safe)
        }
        Intrinsic::ThreadgroupBarrier => declaration(tcx, definition, &[], Unit, Safety::Unsafe),
        Intrinsic::SimdSum => declaration(tcx, definition, &[F32], F32, Safety::Unsafe),
        Intrinsic::SimdShuffle => declaration(tcx, definition, &[F32, U32], F32, Safety::Unsafe),
        _ => false,
    }
}

fn foreign_header(tcx: TyCtxt<'_>, definition: DefId, safety: Safety) -> bool {
    let signature = tcx.fn_sig(definition).instantiate_identity().skip_binder();
    tcx.is_foreign_item(definition)
        && signature.abi().is_rustic_abi()
        && signature.safety() == safety
        && !signature.c_variadic()
        && tcx.generics_of(definition).count() == 0
}

fn declaration(
    tcx: TyCtxt<'_>,
    definition: DefId,
    inputs: &[Slot],
    output: Slot,
    safety: Safety,
) -> bool {
    let signature = tcx.fn_sig(definition).instantiate_identity().skip_binder();
    foreign_header(tcx, definition, safety)
        && signature.inputs().len() == inputs.len()
        && signature
            .inputs()
            .iter()
            .zip(inputs)
            .all(|(&ty, &slot)| slot_matches(tcx, slot, ty))
        && slot_matches(tcx, output, signature.output())
}

fn slot_matches(tcx: TyCtxt<'_>, slot: Slot, ty: Ty<'_>) -> bool {
    match slot {
        Slot::F32 => matches!(ty.kind(), ty::Float(ty::FloatTy::F32)),
        Slot::U32 => matches!(ty.kind(), ty::Uint(ty::UintTy::U32)),
        Slot::Usize => matches!(ty.kind(), ty::Uint(ty::UintTy::Usize)),
        Slot::Unit => ty.is_unit(),
        Slot::Pointer(mutability) => {
            matches!(ty.kind(), ty::RawPtr(pointee, mutbl) if pointee.is_unit() && *mutbl == mutability)
        }
        Slot::Dim3 | Slot::F16 => {
            let ty::Adt(definition, _) = ty.kind() else {
                return false;
            };
            let expected = if matches!(slot, Slot::Dim3) {
                DeviceType::Dim3
            } else {
                DeviceType::F16
            };
            device::device_type(tcx, definition.did()) == Some(expected)
        }
    }
}

fn buffer_element<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<(DeviceType, Ty<'tcx>)> {
    let ty::Adt(definition, arguments) = ty.kind() else {
        return None;
    };
    let kind = device::device_type(tcx, definition.did())?;
    match kind {
        DeviceType::ReadBuffer
        | DeviceType::WriteBuffer
        | DeviceType::AtomicBuffer
        | DeviceType::ThreadgroupBuffer => Some((kind, arguments.first()?.as_type()?)),
        _ => None,
    }
}

fn adapter(tcx: TyCtxt<'_>, definition: DefId, operation: Intrinsic) -> bool {
    let signature = tcx.fn_sig(definition).instantiate_identity().skip_binder();
    if !signature.abi().is_rustic_abi()
        || signature.c_variadic()
        || !tcx.is_mir_available(definition)
    {
        return false;
    }
    let input = signature.inputs();
    let output = signature.output();
    let bridge = match (operation, input) {
        (Intrinsic::ThreadgroupAlloc, []) if signature.safety() == Safety::Safe => {
            let Some((DeviceType::ThreadgroupBuffer, _)) = buffer_element(tcx, output) else {
                return false;
            };
            if !allocation_shape(tcx, definition, output) {
                return false;
            }
            "__metal_threadgroup_alloc"
        }
        (Intrinsic::BufferLoad, &[buffer, index]) if index == tcx.types.u32 => {
            let Some((kind, element)) = buffer_element(tcx, buffer) else {
                return false;
            };
            if output != element {
                return false;
            }
            match kind {
                DeviceType::ReadBuffer => "__metal_buffer_load",
                DeviceType::ThreadgroupBuffer => "__metal_threadgroup_load",
                _ => return false,
            }
        }
        (Intrinsic::BufferStore, &[buffer, index, value])
            if index == tcx.types.u32 && output.is_unit() =>
        {
            let Some((kind, element)) = buffer_element(tcx, buffer) else {
                return false;
            };
            if value != element {
                return false;
            }
            match kind {
                DeviceType::WriteBuffer => "__metal_buffer_store",
                DeviceType::ThreadgroupBuffer => "__metal_threadgroup_store",
                _ => return false,
            }
        }
        (Intrinsic::AtomicAdd, &[buffer, index, value]) if index == tcx.types.u32 => {
            let Some((DeviceType::AtomicBuffer, element)) = buffer_element(tcx, buffer) else {
                return false;
            };
            if value != element || output != element {
                return false;
            }
            "__metal_atomic_add"
        }
        _ => return false,
    };
    if !matches!(operation, Intrinsic::ThreadgroupAlloc) && signature.safety() != Safety::Unsafe {
        return false;
    }
    let body = tcx.instance_mir(ty::InstanceKind::Item(definition));
    let mut declarations = body.basic_blocks.iter().filter_map(|block| {
        let mir::TerminatorKind::Call { func, .. } = &block.terminator().kind else {
            return None;
        };
        let ty::FnDef(callee, _) = *func.ty(&body.local_decls, tcx).kind() else {
            return None;
        };
        tcx.is_foreign_item(callee).then_some(callee)
    });
    let Some(callee) = declarations.next() else {
        return false;
    };
    declarations.next().is_none()
        && callee.krate == definition.krate
        && device::in_module(tcx, callee, "intrinsics")
        && tcx.item_name(callee).as_str() == bridge
        && memory_declaration(tcx, callee, operation)
}

fn allocation_shape<'tcx>(tcx: TyCtxt<'tcx>, definition: DefId, output: Ty<'tcx>) -> bool {
    let generics = tcx.generics_of(definition);
    let [element, length] = generics.own_params.as_slice() else {
        return false;
    };
    if generics.parent_count != 0
        || !matches!(element.kind, ty::GenericParamDefKind::Type { .. })
        || !matches!(length.kind, ty::GenericParamDefKind::Const { .. })
    {
        return false;
    }
    let length_type = tcx.normalize_erasing_regions(
        ty::TypingEnv::non_body_analysis(tcx, definition),
        tcx.type_of(length.def_id).instantiate_identity(),
    );
    let ty::Adt(_, arguments) = output.kind() else {
        return false;
    };
    length_type == tcx.types.usize
        && *arguments == ty::GenericArgs::identity_for_item(tcx, definition)
}

fn memory_declaration(tcx: TyCtxt<'_>, definition: DefId, operation: Intrinsic) -> bool {
    use Slot::*;
    let read = Pointer(Mutability::Not);
    let write = Pointer(Mutability::Mut);
    match operation {
        Intrinsic::BufferLoad => declaration(
            tcx,
            definition,
            &[read, U32, write, Usize],
            Unit,
            Safety::Unsafe,
        ),
        Intrinsic::BufferStore => declaration(
            tcx,
            definition,
            &[write, U32, read, Usize],
            Unit,
            Safety::Unsafe,
        ),
        Intrinsic::ThreadgroupAlloc => {
            declaration(tcx, definition, &[Usize, Usize], write, Safety::Unsafe)
        }
        Intrinsic::AtomicAdd => declaration(
            tcx,
            definition,
            &[write, U32, read, write, Usize],
            Unit,
            Safety::Unsafe,
        ),
        _ => false,
    }
}
