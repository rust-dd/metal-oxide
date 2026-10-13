use rustc_middle::{
    mir,
    ty::{TyCtxt, TypeVisitableExt},
};

use super::Result;
use crate::intrinsics::{PanicEntry, panic_entry};

/// Literal panic messages of one body: `panic("..")` calls, and `Arguments::from_str("..")`
/// values whose call returns directly into `panic_fmt`.
pub(super) struct Panics {
    literals: Vec<Option<String>>,
}

impl Panics {
    /// Rejects formatted panics before their argument locals would fail type lowering.
    pub(super) fn new<'tcx>(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>) -> Result<Self> {
        let mut literals = vec![None; body.local_decls.len()];
        for block in body.basic_blocks.iter() {
            if let mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target: Some(target),
                ..
            } = &block.terminator().kind
                && entry(tcx, func) == Some(PanicEntry::LiteralArguments)
                && let Some(local) = destination.as_local()
                && let mir::TerminatorKind::Call {
                    func: next,
                    args: forwarded,
                    ..
                } = &body.basic_blocks[*target].terminator().kind
                && entry(tcx, next) == Some(PanicEntry::PanicFmt)
                && forwarded[0].node.place().and_then(|place| place.as_local()) == Some(local)
            {
                literals[local.as_usize()] = literal(tcx, &args[0].node);
            }
        }
        let panics = Self { literals };
        for block in body.basic_blocks.iter() {
            let terminator = block.terminator();
            if let mir::TerminatorKind::Call { func, .. } = &terminator.kind
                && entry(tcx, func).is_some()
                && panics.message(tcx, &terminator.kind).is_none()
            {
                let span = terminator.source_info.span;
                return Err((
                    span.ctxt().outer_expn().expansion_cause().unwrap_or(span),
                    "device panics need a literal message; format arguments are unsupported".into(),
                ));
            }
        }
        Ok(panics)
    }

    /// Locals holding a literal `Arguments`; they need no device representation.
    pub(super) fn is_literal(&self, local: usize) -> bool {
        self.literals[local].is_some()
    }

    /// The message of a call that panics; `Arguments::from_str` panics in place of its
    /// `panic_fmt` successor.
    pub(super) fn message<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        kind: &mir::TerminatorKind<'tcx>,
    ) -> Option<String> {
        let mir::TerminatorKind::Call {
            func,
            args,
            destination,
            ..
        } = kind
        else {
            return None;
        };
        let local =
            |place: Option<mir::Place<'tcx>>| self.literals[place?.as_local()?.as_usize()].clone();
        match entry(tcx, func)? {
            PanicEntry::Panic => literal(tcx, &args[0].node),
            PanicEntry::PanicFmt => local(args[0].node.place()),
            PanicEntry::LiteralArguments => local(Some(*destination)),
        }
    }
}

fn entry(tcx: TyCtxt<'_>, func: &mir::Operand<'_>) -> Option<PanicEntry> {
    let (definition, _) = func.const_fn_def()?;
    panic_entry(tcx, definition)
}

/// A `&'static str` constant operand.
pub(super) fn literal<'tcx>(tcx: TyCtxt<'tcx>, operand: &mir::Operand<'tcx>) -> Option<String> {
    let mir::Const::Val(value, ty) = operand.constant()?.const_ else {
        return None;
    };
    if !ty.is_ref() || !ty.peel_refs().is_str() || ty.has_param() {
        return None;
    }
    String::from_utf8(value.try_get_slice_bytes_for_diagnostics(tcx)?.to_vec()).ok()
}
