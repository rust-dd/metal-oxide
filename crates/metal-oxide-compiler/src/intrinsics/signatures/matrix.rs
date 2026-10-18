use metal_oxide_ir::MatrixOp;
use rustc_hir::{Mutability, Safety, def_id::DefId};
use rustc_middle::ty::{self, Ty, TyCtxt};

use super::{Slot, buffer_element};
use crate::intrinsics::device::{self, DeviceType};

/// The bridge an adapter of `op` with these parameter and result types calls.
pub(super) fn adapter<'tcx>(
    tcx: TyCtxt<'tcx>,
    input: &[Ty<'tcx>],
    output: Ty<'tcx>,
    op: MatrixOp,
) -> Option<&'static str> {
    let element = |ty: Ty<'tcx>| match ty.kind() {
        ty::Adt(definition, arguments)
            if device::device_type(tcx, definition.did()) == Some(DeviceType::Matrix) =>
        {
            arguments.first()?.as_type()
        }
        _ => None,
    };
    let positions = |positions: [Ty<'tcx>; 3]| positions.iter().all(|&ty| ty == tcx.types.u32);
    match (op, input) {
        (MatrixOp::Splat, &[value]) => {
            (element(output) == Some(value)).then_some("__metal_matrix_splat")
        }
        (MatrixOp::MultiplyAccumulate, &[a, b, c]) => {
            (element(a).is_some() && element(b).is_some() && element(c).is_some() && output == c)
                .then_some("__metal_matrix_multiply_accumulate")
        }
        (MatrixOp::Load { .. }, &[buffer, row, column, stride]) => {
            let (kind, item) = buffer_element(tcx, buffer)?;
            (kind.readable() && positions([row, column, stride]) && element(output) == Some(item))
                .then_some("__metal_matrix_load")
        }
        (MatrixOp::Store { .. }, &[buffer, row, column, stride, matrix]) => {
            let (kind, item) = buffer_element(tcx, buffer)?;
            (kind.writable()
                && positions([row, column, stride])
                && element(matrix) == Some(item)
                && output.is_unit())
            .then_some("__metal_matrix_store")
        }
        _ => None,
    }
}

/// Whether a matrix bridge takes the pointers, positions and flag its adapters pass.
pub(super) fn bridge(tcx: TyCtxt<'_>, definition: DefId, op: MatrixOp) -> bool {
    let read = Slot::Pointer(Mutability::Not);
    let write = Slot::Pointer(Mutability::Mut);
    let inputs: &[Slot] = match op {
        MatrixOp::Splat => &[read, write, Slot::Usize],
        MatrixOp::MultiplyAccumulate => &[read, read, read, write],
        MatrixOp::Load { .. } => &[read, Slot::U32, Slot::U32, Slot::U32, Slot::Bool, write],
        MatrixOp::Store { .. } => &[write, Slot::U32, Slot::U32, Slot::U32, Slot::Bool, read],
    };
    super::declaration(tcx, definition, inputs, Slot::Unit, Safety::Unsafe)
}
