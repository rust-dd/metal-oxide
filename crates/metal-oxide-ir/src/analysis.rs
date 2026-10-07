use crate::{ControlFlowGraph, Dominators, Expression};

/// Control-flow facts and effects established by IR validation.
#[derive(Debug)]
pub struct FunctionAnalysis {
    pub cfg: ControlFlowGraph,
    pub dominators: Dominators,
    pub postdominators: Dominators,
    pub cooperative: bool,
}

impl Expression {
    pub fn is_cooperative(&self, functions: &[FunctionAnalysis]) -> bool {
        match self {
            Self::ThreadgroupAlloc { .. }
            | Self::ThreadgroupBarrier
            | Self::SimdSum(_)
            | Self::SimdShuffle { .. } => true,
            Self::Call { function, .. } => functions[*function].cooperative,
            _ => false,
        }
    }
}
