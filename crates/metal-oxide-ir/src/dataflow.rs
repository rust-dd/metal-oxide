use std::collections::BTreeSet;

use crate::*;

pub(crate) fn initialized(function: &Function, graph: &ControlFlowGraph) -> Result<(), Error> {
    let n = function.blocks.len();
    let all = (0..function.locals.len()).collect::<BTreeSet<_>>();
    let entry = (1..=function.parameters).collect::<BTreeSet<_>>();
    let mut inputs = vec![all.clone(); n];
    let mut outputs = vec![all; n];
    loop {
        let mut changed = false;
        for (id, block) in function.blocks.iter().enumerate() {
            if !graph.is_reachable(id) {
                continue;
            }
            let input = if id == 0 {
                entry.clone()
            } else {
                let mut intersection = outputs[graph.predecessors(id)[0]].clone();
                for &pred in &graph.predecessors(id)[1..] {
                    intersection.retain(|v| outputs[pred].contains(v));
                }
                intersection
            };
            let mut output = input.clone();
            output.extend(block.statements.iter().map(|s| s.destination));
            changed |= inputs[id] != input || outputs[id] != output;
            inputs[id] = input;
            outputs[id] = output;
        }
        if !changed {
            break;
        }
    }
    for (id, block) in function.blocks.iter().enumerate() {
        if !graph.is_reachable(id) {
            continue;
        }
        let mut initialized = inputs[id].clone();
        for statement in &block.statements {
            for operand in statement.value.operands() {
                check(function, &initialized, operand, &statement.source)?;
            }
            initialized.insert(statement.destination);
        }
        match &block.terminator {
            Terminator::Branch { condition, .. }
            | Terminator::Assert { condition, .. }
            | Terminator::Switch {
                discriminant: condition,
                ..
            } => check(function, &initialized, condition, &block.source)?,
            Terminator::Return if function.locals[0] != Type::Unit && !initialized.contains(&0) => {
                return Err(Error::new(&block.source, "uninitialized return value"));
            }
            _ => {}
        }
    }
    Ok(())
}

fn check(
    function: &Function,
    initialized: &BTreeSet<usize>,
    operand: &Operand,
    source: &SourceLocation,
) -> Result<(), Error> {
    if let Operand::Place { local, .. } = operand
        && function.locals[*local] != Type::Unit
        && !initialized.contains(local)
    {
        return Err(Error::new(
            source,
            format!("read of uninitialized local {local}"),
        ));
    }
    Ok(())
}
