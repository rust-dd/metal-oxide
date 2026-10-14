use std::borrow::Cow;

use rustc_middle::{
    mir::{
        Body, BorrowKind, Const, ConstOperand, ConstValue, Local, LocalDecl, Location, Operand,
        Place, RETURN_PLACE, Rvalue, Statement, StatementKind, TerminatorKind, WithRetag,
        interpret::{GlobalAlloc, Scalar},
        visit::{PlaceContext, Visitor},
    },
    ty::{self, EarlyBinder, Instance, Mutability, Ty, TyCtxt, TypingEnv},
};
use rustc_span::Spanned;

use crate::types::has_reference;

/// Where MIR borrows a constant.
enum Site {
    /// `place = const &C`.
    Assignment(usize),
    /// A call argument `const &C`.
    Argument(usize),
}

/// Borrows of constants, such as `&CONST` or the promoted `&ControlFlow::Continue(())` in core's
/// iterator adapters, become borrows of local copies: kernels have no constant data to point
/// into. Constants that hold pointers, such as panic messages, and borrows that can outlive the
/// function stay as they are, and the importer rejects them.
pub(crate) fn copy<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &mut Cow<'_, Body<'tcx>>,
) {
    let mut sites = Vec::new();
    for (block, data) in body.basic_blocks.iter_enumerated() {
        for (index, statement) in data.statements.iter().enumerate() {
            if let Some((place, Rvalue::Use(Operand::Constant(constant), _))) =
                statement.kind.as_assign()
                && let Some(copied) = copied(tcx, instance, constant)
                && !Flow::new(tcx, instance, body).escapes(|flow| flow.receive(*place))
            {
                sites.push((block, Site::Assignment(index), copied));
            }
        }
        if let TerminatorKind::Call {
            args, destination, ..
        } = &data.terminator().kind
        {
            for (index, argument) in args.iter().enumerate() {
                if let Operand::Constant(constant) = &argument.node
                    && let Some(copied) = copied(tcx, instance, constant)
                    && !Flow::new(tcx, instance, body).escapes(|flow| flow.call(args, *destination))
                {
                    sites.push((block, Site::Argument(index), copied));
                }
            }
        }
    }
    if sites.is_empty() {
        return;
    }
    let body = body.to_mut();
    // Later statements first, so that insertions keep the indices of earlier ones.
    for (block, site, (value, ty)) in sites.into_iter().rev() {
        let span = value.span;
        let copy = body.local_decls.push(LocalDecl::new(ty, span));
        let borrow = Rvalue::Ref(
            tcx.lifetimes.re_erased,
            BorrowKind::Shared,
            Place::from(copy),
        );
        match site {
            Site::Assignment(index) => {
                let statement = &mut body.basic_blocks_mut()[block].statements[index];
                let source_info = statement.source_info;
                let StatementKind::Assign(assignment) = &mut statement.kind else {
                    unreachable!()
                };
                let mut initializer = std::mem::replace(&mut assignment.1, borrow);
                let Rvalue::Use(operand, _) = &mut initializer else {
                    unreachable!()
                };
                *operand = Operand::Constant(Box::new(value));
                let initialize = Statement::new(
                    source_info,
                    StatementKind::Assign(Box::new((Place::from(copy), initializer))),
                );
                body.basic_blocks_mut()[block]
                    .statements
                    .insert(index, initialize);
            }
            Site::Argument(index) => {
                let reference = Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, ty);
                let reference = body.local_decls.push(LocalDecl::new(reference, span));
                let data = &mut body.basic_blocks_mut()[block];
                let source_info = data.terminator().source_info;
                let TerminatorKind::Call { args, .. } = &mut data.terminator_mut().kind else {
                    unreachable!()
                };
                args[index].node = Operand::Move(Place::from(reference));
                let initializer = Rvalue::Use(Operand::Constant(Box::new(value)), WithRetag::No);
                data.statements.extend([
                    Statement::new(
                        source_info,
                        StatementKind::Assign(Box::new((Place::from(copy), initializer))),
                    ),
                    Statement::new(
                        source_info,
                        StatementKind::Assign(Box::new((Place::from(reference), borrow))),
                    ),
                ]);
            }
        }
    }
}

/// The constant that `constant` borrows and its type, when it is a sized value without pointers.
fn copied<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    constant: &ConstOperand<'tcx>,
) -> Option<(ConstOperand<'tcx>, Ty<'tcx>)> {
    let typing = TypingEnv::fully_monomorphized();
    let value = instance.instantiate_mir_and_normalize_erasing_regions(
        tcx,
        typing,
        EarlyBinder::bind(tcx, constant.const_),
    );
    let ty::Ref(_, target, _) = value.ty().kind() else {
        return None;
    };
    if !target.is_sized(tcx, typing) || crate::types::is_zero_sized(tcx, *target) {
        return None;
    }
    let ConstValue::Scalar(Scalar::Ptr(pointer, _)) =
        value.eval(tcx, typing, constant.span).ok()?
    else {
        return None;
    };
    let (provenance, offset) = pointer.into_raw_parts();
    let alloc_id = provenance.alloc_id();
    let GlobalAlloc::Memory(allocation) = tcx.global_alloc(alloc_id) else {
        return None;
    };
    if !allocation.inner().provenance().ptrs().is_empty() {
        return None;
    }
    let copied = ConstOperand {
        span: constant.span,
        user_ty: None,
        const_: Const::Val(ConstValue::Indirect { alloc_id, offset }, *target),
    };
    Some((copied, *target))
}

/// Where a borrow of a copy can go: the locals that hold it, or a value made from it.
struct Flow<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &'a Body<'tcx>,
    holders: Vec<bool>,
    changed: bool,
}

impl<'a, 'tcx> Flow<'a, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>, body: &'a Body<'tcx>) -> Self {
        Self {
            tcx,
            instance,
            body,
            holders: vec![false; body.local_decls.len()],
            changed: false,
        }
    }

    /// Whether the borrow, which `start` hands to its first holders, can outlive the function:
    /// whether it reaches the result, a store through a pointer, or a call that can store it.
    fn escapes(mut self, start: impl FnOnce(&mut Self) -> bool) -> bool {
        if !start(&mut self) {
            return true;
        }
        while std::mem::take(&mut self.changed) {
            for data in self.body.basic_blocks.iter() {
                for statement in &data.statements {
                    if let Some((place, value)) = statement.kind.as_assign() {
                        let mut uses = Uses {
                            holders: &self.holders,
                            found: false,
                        };
                        uses.visit_rvalue(value, Location::START);
                        if uses.found && !self.receive(*place) {
                            return true;
                        }
                    }
                }
                if let TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    ..
                } = &data.terminator().kind
                {
                    let holds = |operand: &Operand<'tcx>| {
                        operand
                            .place()
                            .is_some_and(|place| self.holders[place.local.as_usize()])
                    };
                    let reached = holds(func) || args.iter().any(|argument| holds(&argument.node));
                    if reached && !self.call(args, *destination) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// `place` receives a value made from the borrow; false when that makes it escape. A value
    /// that cannot hold a reference, such as the result of a comparison, does not carry it.
    fn receive(&mut self, place: Place<'tcx>) -> bool {
        let ty = self.monomorphize(place.ty(&self.body.local_decls, self.tcx).ty);
        if !has_reference(self.tcx, ty) {
            return true;
        }
        if place.is_indirect() || place.local == RETURN_PLACE {
            return false;
        }
        let local = place.local.as_usize();
        if !self.holders[local] {
            self.holders[local] = true;
            self.changed = true;
        }
        true
    }

    /// A call that the borrow reaches; false when it can store the borrow through an argument.
    fn call(&mut self, args: &[Spanned<Operand<'tcx>>], destination: Place<'tcx>) -> bool {
        let stores = args.iter().any(|argument| {
            let ty = argument.node.ty(&self.body.local_decls, self.tcx);
            stores(self.tcx, self.monomorphize(ty), 0)
        });
        !stores && self.receive(destination)
    }

    fn monomorphize(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(self.tcx, ty),
        )
    }
}

/// Whether an rvalue reads one of `holders`.
struct Uses<'a> {
    holders: &'a [bool],
    found: bool,
}

impl<'tcx> Visitor<'tcx> for Uses<'_> {
    fn visit_local(&mut self, local: Local, _: PlaceContext, _: Location) {
        self.found |= self.holders[local.as_usize()];
    }
}

/// Whether a value of `ty` gives write access to memory that can hold a reference; very deep
/// types count as such.
fn stores<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, depth: usize) -> bool {
    if depth > 16 {
        return true;
    }
    let inner = |ty: Ty<'tcx>| stores(tcx, ty, depth + 1);
    match ty.kind() {
        ty::Ref(_, target, Mutability::Mut) | ty::RawPtr(target, Mutability::Mut) => {
            has_reference(tcx, *target) || inner(*target)
        }
        ty::Ref(_, target, Mutability::Not) | ty::RawPtr(target, Mutability::Not) => inner(*target),
        ty::Tuple(fields) => fields.iter().any(inner),
        ty::Array(element, _) | ty::Slice(element) => inner(*element),
        ty::Closure(_, arguments) => arguments.as_closure().upvar_tys().iter().any(inner),
        ty::Adt(definition, arguments) => definition.all_fields().any(|field| {
            inner(tcx.normalize_erasing_regions(
                TypingEnv::fully_monomorphized(),
                field.ty(tcx, arguments),
            ))
        }),
        _ => false,
    }
}
