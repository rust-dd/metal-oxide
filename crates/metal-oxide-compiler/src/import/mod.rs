mod calls;
mod control;
mod function;
mod values;

use std::collections::HashMap;

use metal_oxide_ir as ir;
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_span::Span;

use crate::types::TypeLowering;
use function::FunctionImporter;

type Result<T> = std::result::Result<T, (Span, String)>;

pub(crate) struct ModuleImporter<'tcx> {
    tcx: TyCtxt<'tcx>,
    function_ids: HashMap<Instance<'tcx>, usize>,
    types: TypeLowering<'tcx>,
}

impl<'tcx> ModuleImporter<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            function_ids: HashMap::new(),
            types: TypeLowering::new(tcx),
        }
    }

    pub(crate) fn import(
        mut self,
        entries: &[Instance<'tcx>],
        instances: &[Instance<'tcx>],
    ) -> Result<ir::Module> {
        if entries.is_empty() {
            return Err((
                rustc_span::DUMMY_SP,
                "no kernel entrypoints were found".into(),
            ));
        }
        let instances = instances
            .iter()
            .copied()
            .filter(|i| crate::intrinsics::builtin(self.tcx, i.def_id()).is_none())
            .collect::<Vec<_>>();
        self.function_ids = instances
            .iter()
            .enumerate()
            .map(|(id, &i)| (i, id))
            .collect();
        let mut functions = Vec::with_capacity(instances.len());
        for instance in instances {
            let kernel = entries.contains(&instance);
            functions.push(FunctionImporter::new(&mut self, instance).import(kernel)?);
        }
        Ok(ir::Module {
            functions,
            types: self.types.into_types(),
        })
    }
}

fn location(tcx: TyCtxt<'_>, span: Span) -> ir::SourceLocation {
    let position = tcx.sess.source_map().lookup_char_pos(span.lo());
    ir::SourceLocation {
        file: position
            .file
            .name
            .prefer_remapped_unconditionally()
            .to_string(),
        line: position.line as u32,
        column: position.col.0 as u32 + 1,
    }
}
