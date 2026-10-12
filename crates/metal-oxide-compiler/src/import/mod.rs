mod calls;
mod constants;
mod control;
mod function;
mod literals;
mod places;
mod ranges;
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
            functions: called(functions),
            types: self.types.into_types(),
        })
    }
}

/// Keeps kernels and the functions they can call, renumbering the calls. Folded constant
/// branches leave unreachable blocks; their statements never run and are dropped first.
fn called(mut functions: Vec<ir::Function>) -> Vec<ir::Function> {
    for function in &mut functions {
        let reachable = reachable_blocks(function);
        for (block, reachable) in function.blocks.iter_mut().zip(reachable) {
            if !reachable {
                block.statements.clear();
            }
        }
    }
    let callees = |function: &ir::Function| {
        function
            .blocks
            .iter()
            .flat_map(|block| &block.statements)
            .filter_map(|statement| match statement.value {
                ir::Expression::Call { function, .. } => Some(function),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let mut kept = functions.iter().map(|f| f.kernel).collect::<Vec<_>>();
    let mut pending = (0..functions.len())
        .filter(|&id| kept[id])
        .collect::<Vec<_>>();
    while let Some(id) = pending.pop() {
        for callee in callees(&functions[id]) {
            if !std::mem::replace(&mut kept[callee], true) {
                pending.push(callee);
            }
        }
    }
    let mut next = 0;
    let ids = kept
        .iter()
        .map(|&keep| {
            next += usize::from(keep);
            next - usize::from(keep)
        })
        .collect::<Vec<_>>();
    let mut functions = functions
        .into_iter()
        .zip(kept)
        .filter_map(|(function, keep)| keep.then_some(function))
        .collect::<Vec<_>>();
    for statement in functions
        .iter_mut()
        .flat_map(|f| &mut f.blocks)
        .flat_map(|b| &mut b.statements)
    {
        if let ir::Expression::Call { function, .. } = &mut statement.value {
            *function = ids[*function];
        }
    }
    functions
}

fn reachable_blocks(function: &ir::Function) -> Vec<bool> {
    let mut reachable = vec![false; function.blocks.len()];
    let mut pending = vec![0];
    while let Some(id) = pending.pop() {
        if id < reachable.len() && !std::mem::replace(&mut reachable[id], true) {
            pending.extend(function.blocks[id].terminator.successors());
        }
    }
    reachable
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
