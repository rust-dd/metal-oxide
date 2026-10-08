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

pub(crate) fn validate(
    module: &Module,
    analyses: &mut [FunctionAnalysis],
    order: &[usize],
) -> Result<(), Error> {
    let mut summaries = vec![Summary::default(); module.functions.len()];
    let kernels = module.functions.iter().any(|function| function.kernel);
    for &id in order {
        let function = &module.functions[id];
        let analysis = &analyses[id];
        allocations(function, &analysis.cfg)?;
        let (values, control) = flow(function, analysis, &summaries);
        let mut summary = Summary {
            returns: values[0].clone(),
            ..Summary::default()
        };
        let mut cooperative = false;
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
                cooperative = true;
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
