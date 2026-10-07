use std::collections::HashSet;

use rustc_hir::def::DefKind;
use rustc_middle::{
    mir::{Body, Operand, TerminatorKind},
    ty::{self, EarlyBinder, Instance, TyCtxt, TypingEnv},
};
use rustc_span::Span;

pub(crate) fn reachable<'tcx>(
    tcx: TyCtxt<'tcx>,
    entries: &[Instance<'tcx>],
) -> Vec<Instance<'tcx>> {
    let mut collector = Collector {
        tcx,
        seen: HashSet::new(),
        active: HashSet::new(),
        instances: Vec::new(),
    };
    for &entry in entries {
        collector.visit(entry, tcx.def_span(entry.def_id()));
    }
    collector.instances
}

struct Collector<'tcx> {
    tcx: TyCtxt<'tcx>,
    seen: HashSet<Instance<'tcx>>,
    active: HashSet<Instance<'tcx>>,
    instances: Vec<Instance<'tcx>>,
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
        if !matches!(instance.def, ty::InstanceKind::Item(_)) {
            tcx.dcx()
                .span_err(span, format!("unsupported function instance: {instance}"));
            return;
        }
        if !matches!(
            tcx.def_kind(instance.def_id()),
            DefKind::Fn | DefKind::AssocFn
        ) {
            tcx.dcx().span_err(
                span,
                "closures and callable shims are not supported in Metal kernels",
            );
            return;
        }
        let signature = tcx
            .fn_sig(instance.def_id())
            .instantiate(tcx, instance.args)
            .skip_binder();
        if let Some(builtin) = crate::intrinsics::builtin(tcx, instance.def_id()) {
            crate::trace(format_args!("instance: {instance}"));
            crate::trace(format_args!(
                "builtin: {} mir={}",
                builtin.name(),
                if tcx.is_mir_available(instance.def_id()) {
                    "available"
                } else {
                    "unavailable"
                }
            ));
            self.instances.push(instance);
            return;
        }
        if !signature.abi().is_rustic_abi() || tcx.is_foreign_item(instance.def_id()) {
            tcx.dcx()
                .span_err(span, "foreign ABI calls are not supported in Metal kernels");
            return;
        }
        if !tcx.is_mir_available(instance.def_id()) {
            tcx.dcx()
                .span_err(span, format!("device MIR is unavailable for {instance}"));
            return;
        }
        crate::trace(format_args!("instance: {instance}"));
        self.instances.push(instance);
        let body = tcx.instance_mir(instance.def);
        let assertions = body
            .basic_blocks
            .iter()
            .filter(|block| matches!(block.terminator().kind, TerminatorKind::Assert { .. }))
            .count();
        crate::trace(format_args!(
            "mir: blocks={} locals={} asserts={assertions}",
            body.basic_blocks.len(),
            body.local_decls.len()
        ));
        for (local, declaration) in body.local_decls.iter_enumerated() {
            let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(tcx, declaration.ty),
            );
            crate::trace(format_args!("local {local:?}: {ty}"));
        }
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
