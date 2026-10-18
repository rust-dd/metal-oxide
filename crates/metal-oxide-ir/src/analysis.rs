use std::collections::BTreeSet;

use crate::{
    AddressSpace, ControlFlowGraph, Dominators, Expression, Function, Module, Operand, Place,
    Statement, typing::place_info,
};

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
    /// The function or a callee reads the thread index, so its kernel requires a
    /// one-dimensional launch.
    pub one_dimensional: bool,
}

impl Expression {
    pub fn is_cooperative(&self, functions: &[FunctionAnalysis]) -> bool {
        match self {
            Self::ThreadgroupAlloc { .. }
            | Self::ThreadgroupBarrier
            | Self::Simd { .. }
            | Self::Matrix { .. } => true,
            Self::Call { function, .. } => functions[*function].cooperative,
            _ => false,
        }
    }

    /// The index of the called function, if the expression calls one that can fail a check.
    pub fn fallible_callee(&self, functions: &[FunctionAnalysis]) -> Option<usize> {
        match self {
            Self::Call { function, .. } if functions[*function].fallible => Some(*function),
            _ => None,
        }
    }

    pub fn is_fallible(&self, functions: &[FunctionAnalysis]) -> bool {
        self.fallible_callee(functions).is_some()
    }

    /// A dispatch skipped after an earlier failure must not evaluate the expression.
    pub fn has_effect(&self, functions: &[FunctionAnalysis]) -> bool {
        match self {
            Self::BufferLoad { .. }
            | Self::BufferStore { .. }
            | Self::Atomic { .. }
            | Self::ThreadgroupBarrier
            | Self::Simd { .. }
            | Self::Matrix { .. }
            | Self::Log { .. } => true,
            Self::Call { function, .. } => functions[*function].effects,
            _ => false,
        }
    }
}

impl Statement {
    /// Like [`Expression::has_effect`], and also true for an access to device or threadgroup
    /// memory through a pointer.
    pub fn has_effect(
        &self,
        module: &Module,
        function: &Function,
        functions: &[FunctionAnalysis],
    ) -> bool {
        let shared = |place: &Place| {
            place.is_indirect()
                && place_info(module, function, place, &self.source)
                    .is_ok_and(|info| info.space != AddressSpace::Thread)
        };
        let reads = match &self.value {
            Expression::Ref { .. } => Vec::new(),
            value => value.operands(),
        };
        self.value.has_effect(functions)
            || shared(&self.destination)
            || reads
                .into_iter()
                .any(|operand| matches!(operand, Operand::Place(place) if shared(place)))
    }
}
