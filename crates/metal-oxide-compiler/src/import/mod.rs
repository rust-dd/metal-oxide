mod arrays;
mod atomics;
mod buffers;
mod calls;
mod constants;
mod contracts;
mod control;
mod conversions;
mod function;
mod handles;
mod iterators;
mod literals;
mod memory;
mod messages;
mod panics;
mod places;
mod ranges;
mod references;
mod slices;
mod values;
mod vector;

use std::collections::{HashMap, HashSet};

use metal_oxide_ir as ir;
use rustc_middle::ty::{EarlyBinder, Instance, TyCtxt, TypingEnv};
use rustc_span::{Span, def_id::LOCAL_CRATE};

use crate::types::TypeLowering;
use function::FunctionImporter;

type Result<T> = std::result::Result<T, (Span, String)>;

/// What a function is imported for.
#[derive(Clone, PartialEq, Eq, Hash)]
struct FunctionKey<'tcx> {
    instance: Instance<'tcx>,
    entry: Option<String>,
    parameters: Vec<ir::Type>,
    /// The literals of the message parameters.
    messages: Vec<Option<String>>,
    /// Where the checks of a `#[track_caller]` function report.
    caller: Option<ir::SourceLocation>,
    /// The return type, for a function whose return values hold no reference of their own,
    /// such as `None`; its references point where the caller's do.
    returns: Option<ir::Type>,
}

/// Kernel code the importer rejects. A rejection in another crate's code, such as core's,
/// reports at the kernel crate's call that reached it, with the code it reached.
pub(crate) struct Rejection {
    pub(crate) span: Span,
    pub(crate) message: String,
    pub(crate) reached: Option<Span>,
}

pub(crate) struct ModuleImporter<'tcx> {
    tcx: TyCtxt<'tcx>,
    /// Where a rejection that a call reported at itself arose.
    reached: Option<Span>,
    /// The kernel entrypoints, which no function may call.
    entries: HashSet<Instance<'tcx>>,
    ids: HashMap<FunctionKey<'tcx>, usize>,
    /// Empty while the function is being imported.
    functions: Vec<Option<ir::Function>>,
    types: TypeLowering<'tcx>,
}

impl<'tcx> ModuleImporter<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            reached: None,
            entries: HashSet::new(),
            ids: HashMap::new(),
            functions: Vec::new(),
            types: TypeLowering::new(tcx),
        }
    }

    pub(crate) fn import(
        mut self,
        entries: &[crate::collect::Entry<'tcx>],
    ) -> std::result::Result<ir::Module, Rejection> {
        self.functions_of(entries)
            .map_err(|(span, message)| Rejection {
                span,
                message,
                reached: self.reached,
            })?;
        Ok(ir::Module {
            functions: called(self.functions.into_iter().map(Option::unwrap).collect()),
            types: self.types.into_types(),
        })
    }

    /// Imports the entries and every function they call.
    fn functions_of(&mut self, entries: &[crate::collect::Entry<'tcx>]) -> Result<()> {
        if entries.is_empty() {
            return Err((
                rustc_span::DUMMY_SP,
                "no kernel entrypoints were found".into(),
            ));
        }
        self.entries = entries.iter().map(|entry| entry.instance).collect();
        for named in entries {
            let entry = named.instance;
            let body = self.tcx.instance_mir(entry.def);
            let parameters = body
                .args_iter()
                .map(|local| {
                    let declaration = &body.local_decls[local];
                    let ty = entry.instantiate_mir_and_normalize_erasing_regions(
                        self.tcx,
                        TypingEnv::fully_monomorphized(),
                        EarlyBinder::bind(self.tcx, declaration.ty),
                    );
                    self.types.lower(ty, declaration.source_info.span)
                })
                .collect::<Result<Vec<_>>>()?;
            let messages = vec![None; parameters.len()];
            let key = FunctionKey {
                instance: entry,
                entry: Some(named.name.clone()),
                parameters,
                messages,
                caller: None,
                returns: None,
            };
            self.function(key, true)?;
        }
        Ok(())
    }

    /// The function for `key`, imported on first use.
    fn function(&mut self, key: FunctionKey<'tcx>, kernel: bool) -> Result<usize> {
        if let Some(&id) = self.ids.get(&key) {
            return Ok(id);
        }
        let id = self.functions.len();
        self.functions.push(None);
        self.ids.insert(key.clone(), id);
        let entry = key.entry.clone();
        let mut function = FunctionImporter::new(self, key).import(kernel)?;
        if let Some(name) = entry {
            function.name = name;
        }
        self.functions[id] = Some(function);
        Ok(id)
    }

    fn return_type(&self, id: usize) -> ir::Type {
        self.functions[id]
            .as_ref()
            .expect("callees are imported before their callers")
            .locals[0]
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

/// The note on a rejection that a kernel crate call reached in another crate's code.
pub(crate) const REACHED: &str = "in code from another crate";

/// Whether `span` lies in the kernel crate's own source files.
pub(crate) fn local(tcx: TyCtxt<'_>, span: Span) -> bool {
    tcx.sess.source_map().lookup_source_file(span.lo()).cnum == LOCAL_CRATE
}

/// Where `span` is. Code that another crate's macro expands in this crate, such as a kernel's
/// `log!`, is at the macro call.
fn location(tcx: TyCtxt<'_>, span: Span) -> ir::SourceLocation {
    let span = match span.ctxt().outer_expn().expansion_cause() {
        Some(call) if !local(tcx, span) && local(tcx, call) => call,
        _ => span,
    };
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
