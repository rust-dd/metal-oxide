use rustc_middle::{
    mir,
    ty::{self, Ty, TyCtxt},
};

use super::panics::literal;

/// Whether `ty` borrows a `str` or a trait object, at any depth. Neither has a device
/// representation: a message is a literal known when compiling, and trait objects only reach
/// panics.
pub(crate) fn is_message(ty: Ty<'_>) -> bool {
    let mut target = ty;
    while let ty::Ref(_, inner, _) = *target.kind() {
        target = inner;
    }
    target != ty && (target.is_str() || matches!(target.kind(), ty::Dynamic(..)))
}

/// The literal of each local that holds one: a parameter's from the call that specialized the
/// function, another's from the literal or local it copies or borrows.
pub(super) fn messages<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    parameters: &[Option<String>],
) -> Vec<Option<String>> {
    let mut messages = vec![None; body.local_decls.len()];
    for (index, message) in parameters.iter().enumerate() {
        messages[index + 1] = message.clone();
    }
    loop {
        let mut changed = false;
        for statement in body.basic_blocks.iter().flat_map(|block| &block.statements) {
            let mir::StatementKind::Assign(assignment) = &statement.kind else {
                continue;
            };
            let (destination, value) = assignment.as_ref();
            let Some(local) = destination.as_local() else {
                continue;
            };
            if messages[local.as_usize()].is_some() {
                continue;
            }
            let message = match value {
                mir::Rvalue::Use(operand, _) => message(tcx, operand, &messages),
                mir::Rvalue::Ref(_, _, place) | mir::Rvalue::CopyForDeref(place) => {
                    borrowed(place, &messages)
                }
                _ => None,
            };
            if message.is_some() {
                messages[local.as_usize()] = message;
                changed = true;
            }
        }
        if !changed {
            return messages;
        }
    }
}

/// The literal an operand passes: a string constant, or a local that holds one.
pub(super) fn message<'tcx>(
    tcx: TyCtxt<'tcx>,
    operand: &mir::Operand<'tcx>,
    messages: &[Option<String>],
) -> Option<String> {
    match operand {
        mir::Operand::Constant(_) => literal(tcx, operand),
        mir::Operand::Copy(place) | mir::Operand::Move(place) => borrowed(place, messages),
        _ => None,
    }
}

/// The message of a local, or of the local a reborrow dereferences.
fn borrowed(place: &mir::Place<'_>, messages: &[Option<String>]) -> Option<String> {
    match place.projection.as_slice() {
        [] | [mir::ProjectionElem::Deref] => messages[place.local.as_usize()].clone(),
        _ => None,
    }
}
