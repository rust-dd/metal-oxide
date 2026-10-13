use std::collections::BTreeSet;

use crate::*;

#[derive(Clone, Default, PartialEq, Eq)]
struct Dependencies {
    varying: bool,
    parameters: BTreeSet<usize>,
}

impl Dependencies {
    fn merge(&mut self, other: &Self) -> bool {
        let before = self.clone();
        self.varying |= other.varying;
        self.parameters.extend(&other.parameters);
        *self != before
    }

    fn substitute(&self, arguments: &[Dependencies]) -> Self {
        let mut result = Self {
            varying: self.varying,
            parameters: BTreeSet::new(),
        };
        for &parameter in &self.parameters {
            result.merge(&arguments[parameter]);
        }
        result
    }
}

#[derive(Clone, Default)]
struct Summary {
    returns: Dependencies,
    participation: Dependencies,
}

/// Validates uniform participation. A failed check returns from the kernel, so a check that
/// precedes a cooperative operation must be uniform: then every thread fails it together.
pub(crate) fn validate(
    module: &Module,
    analyses: &mut [FunctionAnalysis],
    order: &[usize],
) -> Result<(), Error> {
    for &id in order {
        let function = &module.functions[id];
        allocations(function, &analyses[id].cfg)?;
        let cooperative = analyses[id]
            .cfg
            .reachable_blocks()
            .flat_map(|block| &function.blocks[block].statements)
            .any(|statement| statement.value.is_cooperative(analyses));
        analyses[id].cooperative = cooperative;
    }
    let continuations = continuations(module, analyses, order);
    let mut summaries = vec![Summary::default(); module.functions.len()];
    let kernels = module.functions.iter().any(|function| function.kernel);
    for &id in order {
        let function = &module.functions[id];
        let analysis = &analyses[id];
        let (values, control) = flow(function, analysis, &summaries);
        let mut summary = Summary {
            returns: values[0].clone(),
            ..Summary::default()
        };
        let mut cooperative = analysis.cooperative;
        for &block in &analysis.checks {
            let starts = match &function.blocks[block].terminator {
                Terminator::Assert { target, .. } => vec![*target],
                _ => analysis
                    .cfg
                    .predecessors(block)
                    .iter()
                    .flat_map(|&from| analysis.cfg.successors(from))
                    .copied()
                    .collect(),
            };
            if !reaches_cooperation(function, analysis, analyses, &starts, continuations[id]) {
                continue;
            }
            let mut required = control[block].clone();
            if let Terminator::Assert { condition, .. } = &function.blocks[block].terminator {
                required.merge(&dependencies(condition, &values));
            }
            if required.varying || (!kernels && !required.parameters.is_empty()) {
                return Err(Error::new(
                    &function.blocks[block].source,
                    "a check before a barrier or SIMD operation must be proved or uniform; \
                     a thread that fails it would leave the others waiting",
                ));
            }
            summary.participation.merge(&required);
            cooperative = true;
        }
        for (id, block) in function
            .blocks
            .iter()
            .enumerate()
            .filter(|(id, _)| analysis.cfg.is_reachable(*id))
        {
            for statement in &block.statements {
                if !statement.value.is_cooperative(analyses) {
                    continue;
                }
                let mut required = control[id].clone();
                if let Expression::Call {
                    function,
                    arguments,
                } = &statement.value
                {
                    let arguments = arguments
                        .iter()
                        .map(|v| dependencies(v, &values))
                        .collect::<Vec<_>>();
                    required.merge(&summaries[*function].participation.substitute(&arguments));
                }
                if let Expression::Simd { op, arguments } = &statement.value
                    && op.uniform_control()
                {
                    required.merge(&dependencies(&arguments[1], &values));
                }
                if required.varying || (!kernels && !required.parameters.is_empty()) {
                    return Err(Error::new(
                        &statement.source,
                        "cooperative operation requires uniform participation; divergent control flow is unsupported",
                    ));
                }
                summary.participation.merge(&required);
            }
        }
        summaries[id] = summary;
        analyses[id].cooperative = cooperative;
    }
    Ok(())
}

/// Marks functions after whose return a caller may still run a cooperative operation.
fn continuations(module: &Module, analyses: &[FunctionAnalysis], order: &[usize]) -> Vec<bool> {
    let mut result = vec![false; module.functions.len()];
    for &caller in order.iter().rev() {
        let function = &module.functions[caller];
        let analysis = &analyses[caller];
        for block in analysis.cfg.reachable_blocks() {
            let statements = &function.blocks[block].statements;
            for (index, statement) in statements.iter().enumerate() {
                let Expression::Call {
                    function: callee, ..
                } = statement.value
                else {
                    continue;
                };
                result[callee] |= statements[index + 1..]
                    .iter()
                    .any(|later| later.value.is_cooperative(analyses))
                    || reaches_cooperation(
                        function,
                        analysis,
                        analyses,
                        analysis.cfg.successors(block),
                        result[caller],
                    );
            }
        }
    }
    result
}

/// Whether a path from `starts` runs a cooperative operation, or returns into a caller that may.
fn reaches_cooperation(
    function: &Function,
    analysis: &FunctionAnalysis,
    analyses: &[FunctionAnalysis],
    starts: &[usize],
    continuation: bool,
) -> bool {
    let mut pending = starts.to_vec();
    let mut visited = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !visited.insert(id) {
            continue;
        }
        let block = &function.blocks[id];
        if (continuation && block.terminator == Terminator::Return)
            || block
                .statements
                .iter()
                .any(|statement| statement.value.is_cooperative(analyses))
        {
            return true;
        }
        pending.extend(analysis.cfg.successors(id));
    }
    false
}

fn allocations(function: &Function, graph: &ControlFlowGraph) -> Result<(), Error> {
    let mut ids = BTreeSet::new();
    for (block_id, block) in function
        .blocks
        .iter()
        .enumerate()
        .filter(|(id, _)| graph.is_reachable(*id))
    {
        for statement in &block.statements {
            if let Expression::ThreadgroupAlloc { id, .. } = statement.value {
                if !function.kernel {
                    return Err(Error::new(
                        &statement.source,
                        "threadgroup allocations must be in kernel entrypoints",
                    ));
                }
                if !ids.insert(id) {
                    return Err(Error::new(
                        &statement.source,
                        "duplicate threadgroup allocation ID",
                    ));
                }
                if graph.reaches(block_id, block_id) {
                    return Err(Error::new(
                        &statement.source,
                        "threadgroup allocations inside loops are unsupported",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn flow(
    function: &Function,
    analysis: &FunctionAnalysis,
    summaries: &[Summary],
) -> (Vec<Dependencies>, Vec<Dependencies>) {
    let graph = &analysis.cfg;
    let mut values = vec![Dependencies::default(); function.locals.len()];
    if !function.kernel {
        for (index, value) in values[1..=function.parameters].iter_mut().enumerate() {
            value.parameters.insert(index);
        }
    }
    let mut control = vec![Dependencies::default(); function.blocks.len()];
    loop {
        let mut changed = false;
        for (id, block) in function
            .blocks
            .iter()
            .enumerate()
            .filter(|(id, _)| graph.is_reachable(*id))
        {
            for statement in &block.statements {
                let mut value = control[id].clone();
                for projection in &statement.destination.projection {
                    if let Projection::Index(index) = projection {
                        value.merge(&values[*index]);
                    }
                }
                match &statement.value {
                    Expression::Call {
                        function,
                        arguments,
                    } => {
                        let arguments = arguments
                            .iter()
                            .map(|v| dependencies(v, &values))
                            .collect::<Vec<_>>();
                        value.merge(&summaries[*function].returns.substitute(&arguments));
                    }
                    expression => {
                        value.varying |= matches!(
                            expression,
                            Expression::Coordinates(Builtin::ThreadIdx)
                                | Expression::SimdCoordinate(
                                    SimdBuiltin::Lane | SimdBuiltin::Group
                                )
                                | Expression::BufferLoad { .. }
                                | Expression::Atomic { .. }
                                | Expression::Simd { .. }
                                | Expression::Uninit(_)
                        );
                        for operand in expression.operands() {
                            value.merge(&dependencies(operand, &values));
                        }
                    }
                }
                changed |= values[statement.destination.local].merge(&value);
            }
            if let Terminator::Branch { condition, .. }
            | Terminator::Switch {
                discriminant: condition,
                ..
            } = &block.terminator
            {
                let dependency = dependencies(condition, &values);
                if dependency == Dependencies::default() {
                    continue;
                }
                let join = analysis.postdominators.immediate(id);
                let mut pending = graph.successors(id).to_vec();
                let mut visited = BTreeSet::new();
                while let Some(node) = pending.pop() {
                    if node == join || !visited.insert(node) {
                        continue;
                    }
                    changed |= control[node].merge(&dependency);
                    pending.extend(graph.successors(node));
                }
            }
        }
        if !changed {
            return (values, control);
        }
    }
}

fn dependencies(value: &Operand, values: &[Dependencies]) -> Dependencies {
    match value {
        Operand::Constant(_) => Dependencies::default(),
        Operand::AggregateConstant { fields, .. } => {
            let mut result = Dependencies::default();
            for field in fields {
                result.merge(&dependencies(field, values));
            }
            result
        }
        Operand::Place(place) => {
            let mut result = values[place.local].clone();
            for projection in &place.projection {
                if let Projection::Index(index) = projection {
                    result.merge(&values[*index]);
                }
            }
            result
        }
    }
}
