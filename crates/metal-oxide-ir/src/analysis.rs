use std::collections::BTreeSet;

use crate::{ControlFlowGraph, Dominators, Expression};

/// Control-flow facts and effects established by IR validation.
#[derive(Debug)]
pub struct FunctionAnalysis {
    pub cfg: ControlFlowGraph,
    pub dominators: Dominators,
    pub postdominators: Dominators,
    pub cooperative: bool,
    /// Blocks whose `Assert` or `Panic` terminator needs a runtime check.
    pub checks: BTreeSet<usize>,
    /// The function or a callee can fail a runtime check.
    pub fallible: bool,
    /// The function or a callee accesses buffers, synchronizes, or logs.
    pub effects: bool,
}

impl Expression {
    pub fn is_cooperative(&self, functions: &[FunctionAnalysis]) -> bool {
        match self {
            Self::ThreadgroupAlloc { .. } | Self::ThreadgroupBarrier | Self::Simd { .. } => true,
            Self::Call { function, .. } => functions[*function].cooperative,
            _ => false,
        }
    }

    pub fn is_fallible(&self, functions: &[FunctionAnalysis]) -> bool {
        matches!(self, Self::Call { function, .. } if functions[*function].fallible)
    }

    /// A dispatch skipped after an earlier failure must not evaluate the expression.
    pub fn has_effect(&self, functions: &[FunctionAnalysis]) -> bool {
        match self {
            Self::BufferLoad { .. }
            | Self::BufferStore { .. }
            | Self::Atomic { .. }
            | Self::ThreadgroupBarrier
            | Self::Simd { .. }
            | Self::Log { .. } => true,
            Self::Call { function, .. } => functions[*function].effects,
            _ => false,
        }
    }
}
