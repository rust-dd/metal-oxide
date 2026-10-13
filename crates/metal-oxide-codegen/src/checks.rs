use metal_oxide_artifact as artifact;
use metal_oxide_ir::{Check, FunctionAnalysis, Module, OverflowOp, Terminator};
use std::collections::HashMap;

/// Runtime check sites in emission order; a site's index is its id in the artifact table.
#[derive(Debug)]
pub(crate) struct CheckTable {
    ids: HashMap<(usize, usize), u32>,
    pub(crate) entries: Vec<artifact::Check>,
}

impl CheckTable {
    pub(crate) fn new(module: &Module, analyses: &[FunctionAnalysis]) -> Self {
        let mut ids = HashMap::new();
        let mut entries = Vec::new();
        for (function_id, (function, analysis)) in module.functions.iter().zip(analyses).enumerate()
        {
            for &block_id in &analysis.checks {
                let block = &function.blocks[block_id];
                let kind = match &block.terminator {
                    Terminator::Assert { check, .. } => kind(check),
                    Terminator::Panic(message) => artifact::CheckKind::Panic(message.clone()),
                    _ => unreachable!("runtime checks are assertions or panics"),
                };
                ids.insert((function_id, block_id), entries.len() as u32);
                entries.push(artifact::Check {
                    kind,
                    location: artifact::Location {
                        file: block.source.file.clone(),
                        line: block.source.line,
                        column: block.source.column,
                    },
                });
            }
        }
        Self { ids, entries }
    }

    pub(crate) fn id(&self, function: usize, block: usize) -> u32 {
        self.ids[&(function, block)]
    }
}

fn kind(check: &Check) -> artifact::CheckKind {
    match check {
        Check::Bounds => artifact::CheckKind::Bounds,
        Check::Overflow(op) => artifact::CheckKind::Overflow(match op {
            OverflowOp::Add => artifact::OverflowOp::Add,
            OverflowOp::Sub => artifact::OverflowOp::Sub,
            OverflowOp::Mul => artifact::OverflowOp::Mul,
            OverflowOp::Div => artifact::OverflowOp::Div,
            OverflowOp::Rem => artifact::OverflowOp::Rem,
            OverflowOp::Neg => artifact::OverflowOp::Neg,
            OverflowOp::Shl => artifact::OverflowOp::Shl,
            OverflowOp::Shr => artifact::OverflowOp::Shr,
        }),
        Check::DivisionByZero => artifact::CheckKind::DivisionByZero,
        Check::RemainderByZero => artifact::CheckKind::RemainderByZero,
        Check::SliceStart => artifact::CheckKind::SliceStart,
        Check::SliceEnd => artifact::CheckKind::SliceEnd,
        Check::SliceOrder => artifact::CheckKind::SliceOrder,
    }
}

/// Stores `check + 2` in the submission's failure word unless the dispatch is skipped; when
/// threads fail different checks, one of them remains.
pub(crate) const FAIL: &str = "static void metal_oxide_fail(metal_oxide_context ctx, uint check) {
    *ctx.failed = true;
    if (*ctx.gate == 0u) {
        atomic_store_explicit(ctx.failure, check + 2u, memory_order_relaxed);
    }
}

";
