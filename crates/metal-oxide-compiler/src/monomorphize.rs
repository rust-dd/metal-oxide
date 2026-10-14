use std::{borrow::Cow, collections::HashSet};

use rustc_hir::def::DefKind;
use rustc_middle::{
    mir::{BasicBlock, Body, Operand, Rvalue, StatementKind, TerminatorKind, traversal},
    ty::{self, EarlyBinder, Instance, TyCtxt, TypingEnv},
};
use rustc_span::{Span, sym};

/// Reports recursion and calls the importer cannot lower in every instance the kernels reach.
pub(crate) fn validate<'tcx>(tcx: TyCtxt<'tcx>, entries: &[Instance<'tcx>]) {
    let mut collector = Collector {
        tcx,
        seen: HashSet::new(),
        active: HashSet::new(),
    };
    for &entry in entries {
        collector.visit(entry, tcx.def_span(entry.def_id()));
    }
}

/// The instance's MIR without `cold_path` hints and without the blocks that its constants make
/// unreachable, such as the branches of `T::IS_ZST` or of disabled runtime and precondition
/// checks; codegen skips them likewise.
pub(crate) fn body<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> &'tcx Body<'tcx> {
    let mut body = Cow::Borrowed(tcx.instance_mir(instance.def));
    let gotos = body
        .basic_blocks
        .indices()
        .flat_map(|block| folded(tcx, &body, block))
        .collect::<Vec<_>>();
    if !gotos.is_empty() {
        let blocks = body.to_mut().basic_blocks_mut();
        for (block, target) in gotos {
            blocks[block].terminator_mut().kind = TerminatorKind::Goto { target };
        }
    }
    let reachable = traversal::mono_reachable_as_bitset(&body, tcx, instance);
    let dead = doomed(&body);
    let unreachable = body
        .basic_blocks
        .indices()
        .filter(|&block| !reachable.contains(block) || dead[block.as_usize()])
        .collect::<Vec<_>>();
    if !unreachable.is_empty() {
        let blocks = body.to_mut().basic_blocks_mut();
        for block in unreachable {
            blocks[block].statements.clear();
            blocks[block].terminator_mut().kind = TerminatorKind::Unreachable;
        }
    }
    match body {
        Cow::Borrowed(body) => body,
        Cow::Owned(body) => tcx.arena.alloc(body),
    }
}

/// Blocks from which every path reaches `unreachable`; running them is undefined behavior, as
/// in the `None` arm of `unwrap_unchecked`.
fn doomed(body: &Body<'_>) -> Vec<bool> {
    let mut doomed = vec![false; body.basic_blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for (block, data) in body.basic_blocks.iter_enumerated() {
            let next = match &data.terminator().kind {
                TerminatorKind::Unreachable => true,
                TerminatorKind::Goto { target } => doomed[target.as_usize()],
                TerminatorKind::SwitchInt { targets, .. } => targets
                    .all_targets()
                    .iter()
                    .all(|target| doomed[target.as_usize()]),
                _ => false,
            };
            if next && !doomed[block.as_usize()] {
                doomed[block.as_usize()] = true;
                changed = true;
            }
        }
    }
    doomed
}

/// The `goto`s that replace what is fixed for every instance: a `cold_path` hint, and a switch
/// on a runtime-check setting, read directly or through `check_language_ub`.
fn folded<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    block: BasicBlock,
) -> Vec<(BasicBlock, BasicBlock)> {
    let data = &body.basic_blocks[block];
    match &data.terminator().kind {
        TerminatorKind::Call {
            func,
            destination,
            target: Some(target),
            ..
        } => {
            let Some((definition, _)) = func.const_fn_def() else {
                return Vec::new();
            };
            if tcx.is_intrinsic(definition, sym::cold_path) {
                return vec![(block, *target)];
            }
            if tcx.def_path_str(definition) != "core::ub_checks::check_language_ub" {
                return Vec::new();
            }
            match &body.basic_blocks[*target].terminator().kind {
                TerminatorKind::SwitchInt { discr, targets }
                    if discr.place() == Some(*destination)
                        && body.basic_blocks.predecessors()[*target].len() == 1 =>
                {
                    let enabled = u128::from(tcx.sess.ub_checks());
                    vec![
                        (block, *target),
                        (*target, targets.target_for_value(enabled)),
                    ]
                }
                _ => Vec::new(),
            }
        }
        TerminatorKind::SwitchInt { discr, targets } => {
            let last = data.statements.iter().rev().find(|statement| {
                !matches!(
                    statement.kind,
                    StatementKind::StorageLive(_) | StatementKind::StorageDead(_)
                )
            });
            match last.and_then(|statement| statement.kind.as_assign()) {
                Some((place, Rvalue::Use(Operand::RuntimeChecks(checks), _)))
                    if discr.place() == Some(*place) =>
                {
                    let enabled = u128::from(checks.value(tcx.sess));
                    vec![(block, targets.target_for_value(enabled))]
                }
                _ => Vec::new(),
            }
        }
        _ => Vec::new(),
    }
}

struct Collector<'tcx> {
    tcx: TyCtxt<'tcx>,
    seen: HashSet<Instance<'tcx>>,
    active: HashSet<Instance<'tcx>>,
}

impl<'tcx> Collector<'tcx> {
    fn visit(&mut self, instance: Instance<'tcx>, span: Span) {
        let tcx = self.tcx;
        if self.active.contains(&instance) {
            tcx.dcx()
                .span_err(span, "recursion is not supported in Metal kernels");
            return;
        }
        if !tcx.recursion_limit().value_within_limit(self.active.len()) {
            tcx.dcx().span_err(
                span,
                "device instance depth exceeds the rustc recursion limit",
            );
            return;
        }
        if !self.seen.insert(instance) {
            return;
        }
        // Fn items called through the `Fn` traits run their call shim.
        let fn_item_shim = matches!(
            instance.def,
            ty::InstanceKind::Shim(ty::ShimKind::FnPtr(_, ty)) if matches!(ty.kind(), ty::FnDef(..))
        );
        if !matches!(
            instance.def,
            ty::InstanceKind::Item(_) | ty::InstanceKind::Shim(ty::ShimKind::ClosureOnce { .. })
        ) && !fn_item_shim
            && !(matches!(instance.def, ty::InstanceKind::Intrinsic(_))
                && crate::intrinsics::builtin(tcx, instance.def_id()).is_some())
        {
            tcx.dcx()
                .span_err(span, format!("unsupported function instance: {instance}"));
            return;
        }
        if !matches!(
            tcx.def_kind(instance.def_id()),
            DefKind::Fn | DefKind::AssocFn | DefKind::Closure
        ) {
            tcx.dcx().span_err(
                span,
                "coroutines and other callable shims are not supported in Metal kernels",
            );
            return;
        }
        if let Some(builtin) = crate::intrinsics::builtin(tcx, instance.def_id()) {
            crate::trace(|| {
                println!("instance: {instance}");
                println!(
                    "builtin: {} mir={}",
                    builtin.name(),
                    if tcx.is_mir_available(instance.def_id()) {
                        "available"
                    } else {
                        "unavailable"
                    }
                );
            });
            return;
        }
        // Closures always use the Rust call ABI; `fn_sig` is only defined for other functions.
        let ty = instance.ty(tcx, TypingEnv::fully_monomorphized());
        let rustic = ty.is_closure() || ty.fn_sig(tcx).skip_binder().abi().is_rustic_abi();
        if !rustic || tcx.is_foreign_item(instance.def_id()) {
            tcx.dcx()
                .span_err(span, "foreign ABI calls are not supported in Metal kernels");
            return;
        }
        if matches!(instance.def, ty::InstanceKind::Item(_))
            && !tcx.is_mir_available(instance.def_id())
        {
            tcx.dcx()
                .span_err(span, format!("device MIR is unavailable for {instance}"));
            return;
        }
        let body = body(tcx, instance);
        crate::trace(|| {
            println!("instance: {instance}");
            let assertions = body
                .basic_blocks
                .iter()
                .filter(|block| matches!(block.terminator().kind, TerminatorKind::Assert { .. }))
                .count();
            println!(
                "mir: blocks={} locals={} asserts={assertions}",
                body.basic_blocks.len(),
                body.local_decls.len()
            );
            for (local, declaration) in body.local_decls.iter_enumerated() {
                let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(tcx, declaration.ty),
                );
                println!("local {local:?}: {ty}");
            }
        });
        self.active.insert(instance);
        for block in body.basic_blocks.iter() {
            let terminator = block.terminator();
            match &terminator.kind {
                TerminatorKind::Call { func, .. } | TerminatorKind::TailCall { func, .. } => {
                    self.call(instance, body, func, terminator.source_info.span);
                }
                TerminatorKind::Drop { place, .. } => {
                    let environment = TypingEnv::fully_monomorphized();
                    let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                        tcx,
                        environment,
                        EarlyBinder::bind(tcx, place.ty(&body.local_decls, tcx).ty),
                    );
                    if ty.needs_drop(tcx, environment) {
                        tcx.dcx().span_err(
                            terminator.source_info.span,
                            "destructors are not supported in Metal kernels",
                        );
                    }
                }
                _ => {}
            }
        }
        self.active.remove(&instance);
    }

    fn call(
        &mut self,
        caller: Instance<'tcx>,
        body: &Body<'tcx>,
        operand: &Operand<'tcx>,
        span: Span,
    ) {
        let tcx = self.tcx;
        let environment = TypingEnv::fully_monomorphized();
        let callee = caller.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            environment,
            EarlyBinder::bind(tcx, operand.ty(&body.local_decls, tcx)),
        );
        let ty::FnDef(definition, arguments) = *callee.kind() else {
            tcx.dcx()
                .span_err(span, "indirect calls are not supported in Metal kernels");
            return;
        };
        let arguments = tcx.instantiate_bound_regions_with_erased(arguments);
        if crate::intrinsics::panic_entry(tcx, definition).is_some() {
            return;
        }
        match Instance::try_resolve(tcx, environment, definition, arguments) {
            Ok(Some(instance)) => self.visit(instance, span),
            Ok(None) => {
                tcx.dcx().span_err(
                    span,
                    format!("cannot resolve a concrete device instance for {callee}"),
                );
            }
            Err(_) => {}
        }
    }
}
