use rustc_middle::{
    mir::{
        self,
        visit::{MutatingUseContext, PlaceContext, Visitor},
    },
    ty::{self, Ty, TyCtxt},
};

use crate::intrinsics::{DeviceType, Expanded, device_type};

/// Locals bound once to `&local` or `&mut local`, directly or by reborrow, that stand for
/// `local`: device buffer handles in their method calls, and ranges and chunk iterators whose
/// borrows only advance them through the `next` calls the importer expands.
pub(super) fn handles<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    normalize: impl Fn(Ty<'tcx>) -> Ty<'tcx>,
) -> Vec<Option<mir::Local>> {
    let iterator = |local: mir::Local| Expanded::of(tcx, normalize(body.local_decls[local].ty));
    let handle = |local: mir::Local| {
        iterator(local).is_some()
            || matches!(normalize(body.local_decls[local].ty).kind(),
            ty::Adt(definition, _) if device_type(tcx, definition.did())
                .is_some_and(|kind| {
                    !matches!(kind, DeviceType::Dim3 | DeviceType::F16 | DeviceType::BF16 | DeviceType::ThreadIndex | DeviceType::Matrix)
                }))
    };
    let mut bindings = vec![0_u32; body.local_decls.len()];
    let mut handles = vec![None; body.local_decls.len()];
    for statement in body.basic_blocks.iter().flat_map(|block| &block.statements) {
        let mir::StatementKind::Assign(assignment) = &statement.kind else {
            continue;
        };
        let (destination, value) = assignment.as_ref();
        let Some(local) = destination.as_local() else {
            continue;
        };
        bindings[local.as_usize()] += 1;
        if let mir::Rvalue::Ref(_, _, referent) = value {
            handles[local.as_usize()] = match referent.as_ref() {
                mir::PlaceRef {
                    local,
                    projection: [],
                } if handle(local) => Some(local),
                mir::PlaceRef {
                    local,
                    projection: [mir::ProjectionElem::Deref],
                } => handles[local.as_usize()],
                _ => None,
            };
        }
    }
    for (handle, count) in handles.iter_mut().zip(bindings) {
        if count != 1 {
            *handle = None;
        }
    }
    let mut uses = Uses {
        tcx,
        body,
        normalize: &normalize,
        handles: &handles,
        misused: vec![false; handles.len()],
    };
    uses.visit_body(body);
    let misused = uses.misused;
    for handle in &mut handles {
        if handle.is_some_and(|local| iterator(local).is_some() && misused[local.as_usize()]) {
            *handle = None;
        }
    }
    handles
}

/// Marks the iterators a borrow of which does more than advance them through expanded `next`
/// calls or reborrow into another such borrow.
struct Uses<'a, 'tcx, N> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    normalize: &'a N,
    handles: &'a [Option<mir::Local>],
    misused: Vec<bool>,
}

impl<'tcx, N: Fn(Ty<'tcx>) -> Ty<'tcx>> Visitor<'tcx> for Uses<'_, 'tcx, N> {
    fn visit_local(&mut self, local: mir::Local, context: PlaceContext, location: mir::Location) {
        let Some(target) = self.handles[local.as_usize()] else {
            return;
        };
        if matches!(
            context,
            PlaceContext::NonUse(_) | PlaceContext::MutatingUse(MutatingUseContext::Store)
        ) || self.advances(local, location)
        {
            return;
        }
        self.misused[target.as_usize()] = true;
    }
}

impl<'tcx, N: Fn(Ty<'tcx>) -> Ty<'tcx>> Uses<'_, 'tcx, N> {
    fn advances(&self, local: mir::Local, location: mir::Location) -> bool {
        let block = &self.body.basic_blocks[location.block];
        let Some(statement) = block.statements.get(location.statement_index) else {
            let mir::TerminatorKind::Call { func, args, .. } = &block.terminator().kind else {
                return false;
            };
            let ty::FnDef(definition, arguments) =
                *(self.normalize)(func.ty(&self.body.local_decls, self.tcx)).kind()
            else {
                return false;
            };
            let arguments = self.tcx.instantiate_bound_regions_with_erased(arguments);
            return args.first().and_then(|argument| argument.node.place())
                == Some(mir::Place::from(local))
                && crate::intrinsics::expanded_next(self.tcx, definition, arguments).is_some();
        };
        matches!(
            statement.kind.as_assign(),
            Some((destination, mir::Rvalue::Ref(_, _, place)))
                if place.local == local
                    && destination
                        .as_local()
                        .is_some_and(|borrow| self.handles[borrow.as_usize()] == self.handles[local.as_usize()])
        )
    }
}
