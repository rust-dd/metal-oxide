use crate::control::StructuredGraph;
use metal_oxide_ir::{Function, FunctionAnalysis, Module};

/// Tests the gate word: a skipped dispatch leaves before it touches memory or synchronizes.
pub(crate) const GATE: &str = "if (*metal_oxide_gate != 0u) { return; }";

/// Where a kernel tests its gate: once, at the latest point that dominates every effect, and
/// outside loops. Each extra gate site, or one at kernel entry, costs occupancy on Apple GPUs.
#[derive(Default)]
pub(crate) struct Gates {
    /// The block and statement index the gate precedes.
    statement: Option<(usize, usize)>,
    /// The loop header the gate precedes.
    loop_header: Option<usize>,
}

impl Gates {
    pub(crate) fn new(
        module: &Module,
        function: &Function,
        id: usize,
        analyses: &[FunctionAnalysis],
        graph: &StructuredGraph<'_>,
    ) -> Self {
        let mut gates = Self::default();
        if !function.kernel {
            return gates;
        }
        let analysis = &analyses[id];
        let first_effect = |block: usize| {
            function.blocks[block]
                .statements
                .iter()
                .position(|statement| statement.has_effect(module, function, analyses))
        };
        let effects = analysis
            .cfg
            .reachable_blocks()
            .filter(|&block| first_effect(block).is_some())
            .collect::<Vec<_>>();
        if effects.is_empty() {
            return gates;
        }
        let dominator = analysis.dominators.closest_common(&effects);
        let outermost = graph
            .loops
            .iter()
            .filter(|(header, loop_)| **header == dominator || loop_.members.contains(&dominator))
            .max_by_key(|(_, loop_)| loop_.members.len());
        match outermost {
            Some((&header, _)) => gates.loop_header = Some(header),
            None => {
                let statements = function.blocks[dominator].statements.len();
                gates.statement = Some((dominator, first_effect(dominator).unwrap_or(statements)));
            }
        }
        gates
    }

    pub(crate) fn before_statement(&self, block: usize, statement: usize) -> bool {
        self.statement == Some((block, statement))
    }

    pub(crate) fn before_loop(&self, header: usize) -> bool {
        self.loop_header == Some(header)
    }
}
