use std::collections::BTreeSet;

use crate::*;

pub(crate) fn validate(module: &Module, graphs: &[ControlFlowGraph]) -> Result<(), Error> {
    let mut cooperative = vec![false; module.functions.len()];
    loop {
        let mut changed = false;
        for (id, function) in module.functions.iter().enumerate() {
            let effect = function.blocks.iter().flat_map(|b| &b.statements).any(|s| {
                matches!(
                    s.value,
                    Expression::ThreadgroupBarrier
                        | Expression::ThreadgroupAlloc { .. }
                        | Expression::SimdSum(_)
                        | Expression::SimdShuffle { .. }
                ) || matches!(s.value, Expression::Call { function, .. } if cooperative[function])
            });
            changed |= effect && !cooperative[id];
            cooperative[id] |= effect;
        }
        if !changed {
            break;
        }
    }
    for (function, graph) in module.functions.iter().zip(graphs) {
        check_function(function, graph, &cooperative)?;
    }
    Ok(())
}

fn check_function(
    function: &Function,
    graph: &ControlFlowGraph,
    cooperative: &[bool],
) -> Result<(), Error> {
    let n = function.blocks.len();
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
    let postdominators = graph.postdominators();
    let mut varying = vec![false; function.locals.len()];
    if !function.kernel {
        varying[1..=function.parameters].fill(true);
    }
    let mut divergent = vec![false; n];
    loop {
        let mut changed = false;
        for (id, block) in function
            .blocks
            .iter()
            .enumerate()
            .filter(|(id, _)| graph.is_reachable(*id))
        {
            for statement in &block.statements {
                let variable = divergent[id]
                    || matches!(
                        statement.value,
                        Expression::Coordinates(Builtin::ThreadIdx)
                            | Expression::SimdCoordinate(SimdBuiltin::Lane | SimdBuiltin::Group)
                            | Expression::BufferLoad { .. }
                            | Expression::AtomicAdd { .. }
                            | Expression::Call { .. }
                    )
                    || statement
                        .value
                        .operands()
                        .into_iter()
                        .any(|v| operand_varying(v, &varying));
                if variable && !varying[statement.destination] {
                    varying[statement.destination] = true;
                    changed = true;
                }
            }
            if let Terminator::Branch { condition, .. }
            | Terminator::Switch {
                discriminant: condition,
                ..
            } = &block.terminator
                && operand_varying(condition, &varying)
            {
                let join = postdominators.immediate(id);
                let mut pending = graph.successors(id).to_vec();
                let mut visited = BTreeSet::new();
                while let Some(node) = pending.pop() {
                    if node == join || !visited.insert(node) {
                        continue;
                    }
                    if !divergent[node] {
                        divergent[node] = true;
                        changed = true;
                    }
                    pending.extend(graph.successors(node));
                }
            }
        }
        if !changed {
            break;
        }
    }
    for (id, block) in function
        .blocks
        .iter()
        .enumerate()
        .filter(|(id, _)| graph.is_reachable(*id))
    {
        for statement in &block.statements {
            let synchronized = matches!(
                statement.value,
                Expression::ThreadgroupBarrier
                    | Expression::ThreadgroupAlloc { .. }
                    | Expression::SimdSum(_)
                    | Expression::SimdShuffle { .. }
            ) || matches!(statement.value, Expression::Call { function, .. } if cooperative[function]);
            if synchronized && divergent[id] {
                return Err(Error::new(
                    &statement.source,
                    "cooperative operation requires uniform participation; divergent control flow is unsupported",
                ));
            }
        }
    }
    Ok(())
}

fn operand_varying(value: &Operand, varying: &[bool]) -> bool {
    match value {
        Operand::Constant(_) => false,
        Operand::Place { local, .. } => varying[*local],
    }
}
