use rustc_hir::{ExprKind, StmtKind, def::DefKind, def_id::LocalDefId};
use rustc_middle::ty::{self, Instance, TyCtxt, TypingEnv};
use rustc_span::Symbol;

use super::Entry;

pub(super) fn collect<'tcx>(tcx: TyCtxt<'tcx>, definitions: &[LocalDefId]) -> Vec<Entry<'tcx>> {
    let path = [
        Symbol::intern("metal_oxide_compiler"),
        Symbol::intern("instance"),
    ];
    let mut entries = Vec::new();
    for &definition in definitions {
        let Some(attribute) = tcx.get_attrs_by_path(definition.to_def_id(), &path).next() else {
            continue;
        };
        let Some(name) = attribute.value_str() else {
            tcx.dcx()
                .span_err(tcx.def_span(definition), "invalid kernel instance marker");
            continue;
        };
        match instance(tcx, definition) {
            Some(instance) => entries.push(Entry {
                name: name.to_string(),
                instance,
            }),
            None => {
                tcx.dcx().span_err(
                    tcx.def_span(definition),
                    "kernel instances must name a generic #[kernel] function",
                );
            }
        }
    }
    entries
}

/// The macro marks a type-checked function item, so rustc resolves arguments and trait bounds.
fn instance<'tcx>(tcx: TyCtxt<'tcx>, marker: LocalDefId) -> Option<Instance<'tcx>> {
    if tcx.def_kind(marker) != DefKind::Const {
        return None;
    }
    let body = tcx.hir_body_owned_by(marker);
    let ExprKind::Block(block, _) = body.value.kind else {
        return None;
    };
    let [statement] = block.stmts else {
        return None;
    };
    let StmtKind::Let(binding) = statement.kind else {
        return None;
    };
    let ty = tcx.typeck(marker).expr_ty(binding.init?);
    let ty::FnDef(definition, arguments) = *ty.kind() else {
        return None;
    };
    let path = [
        Symbol::intern("metal_oxide_compiler"),
        Symbol::intern("kernel"),
    ];
    if !definition.is_local()
        || tcx.def_kind(definition) != DefKind::Fn
        || tcx.generics_of(definition).count() == 0
        || tcx.get_attrs_by_path(definition, &path).next().is_none()
    {
        return None;
    }
    let arguments = tcx.instantiate_bound_regions_with_erased(arguments);
    Instance::try_resolve(tcx, TypingEnv::fully_monomorphized(), definition, arguments)
        .ok()
        .flatten()
}
