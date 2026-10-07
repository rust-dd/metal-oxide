use std::collections::BTreeSet;

use crate::*;

pub(crate) fn validate(module: &Module) -> Result<(), Error> {
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
    for function in &module.functions {
        check_function(function, &cooperative)?;
    }
    Ok(())
}

fn check_function(function: &Function, cooperative: &[bool]) -> Result<(), Error> {
    let n = function.blocks.len();
    let successors = function
        .blocks
        .iter()
        .map(|b| b.terminator.successors())
        .collect::<Vec<_>>();
    let mut reachable = vec![false; n];
    let mut pending = vec![0];
    while let Some(id) = pending.pop() {
        if !std::mem::replace(&mut reachable[id], true) {
            pending.extend(&successors[id]);
        }
    }
    let mut ids = BTreeSet::new();
    for (block_id, block) in function
        .blocks
        .iter()
        .enumerate()
        .filter(|(id, _)| reachable[*id])
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
                if reaches(&successors, block_id, block_id) {
                    return Err(Error::new(
                        &statement.source,
                        "threadgroup allocations inside loops are unsupported",
                    ));
                }
            }
        }
    }
    let joins = joins(&successors, &reachable);
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
            .filter(|(id, _)| reachable[*id])
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
                let mut pending = successors[id].clone();
                let mut visited = BTreeSet::new();
                while let Some(node) = pending.pop() {
                    if node == joins[id] || !visited.insert(node) {
                        continue;
                    }
                    if !divergent[node] {
                        divergent[node] = true;
                        changed = true;
                    }
                    pending.extend(&successors[node]);
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
        .filter(|(id, _)| reachable[*id])
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

fn reaches(successors: &[Vec<usize>], from: usize, target: usize) -> bool {
    let mut visited = BTreeSet::new();
    let mut pending = successors[from].clone();
    while let Some(id) = pending.pop() {
        if id == target {
            return true;
        }
        if visited.insert(id) {
            pending.extend(&successors[id]);
        }
    }
    false
}

fn joins(successors: &[Vec<usize>], reachable: &[bool]) -> Vec<usize> {
    let n = successors.len();
    let all = (0..=n)
        .filter(|&id| id == n || reachable[id])
        .collect::<BTreeSet<_>>();
    let mut post = vec![all.clone(); n + 1];
    post[n] = BTreeSet::from([n]);
    loop {
        let mut changed = false;
        for id in (0..n).rev().filter(|&id| reachable[id]) {
            let mut next = all.clone();
            let targets = if successors[id].is_empty() {
                vec![n]
            } else {
                successors[id].clone()
            };
            for target in targets {
                next.retain(|v| post[target].contains(v));
            }
            next.insert(id);
            changed |= next != post[id];
            post[id] = next;
        }
        if !changed {
            break;
        }
    }
    (0..n)
        .map(|id| {
            post[id]
                .iter()
                .copied()
                .filter(|&v| v != id)
                .max_by_key(|&v| post[v].len())
                .unwrap_or(n)
        })
        .collect()
}
