use std::collections::HashSet;

use rustc_hir::def::DefKind;
use rustc_middle::{
    mir::{Body, Operand, TerminatorKind},
    ty::{self, EarlyBinder, Instance, TyCtxt, TypingEnv},
};
use rustc_span::Span;

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
        if !matches!(
            instance.def,
            ty::InstanceKind::Item(_) | ty::InstanceKind::Shim(ty::ShimKind::ClosureOnce { .. })
        ) && !(matches!(instance.def, ty::InstanceKind::Intrinsic(_))
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
        if !tcx.is_mir_available(instance.def_id()) {
            tcx.dcx()
                .span_err(span, format!("device MIR is unavailable for {instance}"));
            return;
        }
        let body = tcx.instance_mir(instance.def);
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
        if tcx.lang_items().next_fn() == Some(definition) {
            if crate::intrinsics::range_next(tcx, definition, arguments).is_none()
                && crate::intrinsics::slice_next(tcx, definition, arguments).is_none()
            {
                tcx.dcx().span_err(
                    span,
                    "iterators other than integer ranges and slice iterators are not supported in Metal kernels",
                );
            }
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
