use std::collections::BTreeSet;

use crate::*;

type State = BTreeSet<Place>;

fn intersect(a: &State, b: &State) -> State {
    let mut result = State::new();
    for left in a {
        for right in b {
            if left.contains(right) {
                result.insert(right.clone());
            } else if right.contains(left) {
                result.insert(left.clone());
            }
        }
    }
    result
}

fn covered(state: &State, place: &Place) -> bool {
    let mut required = place.clone();
    if let Some(index) = required
        .projection
        .iter()
        .position(|p| matches!(p, Projection::Index(_)))
    {
        required.projection.truncate(index);
    }
    state.iter().any(|value| value.contains(&required))
}

fn mark(module: &Module, function: &Function, state: &mut State, place: &Place) {
    if place
        .projection
        .iter()
        .any(|p| matches!(p, Projection::Index(_)))
    {
        return;
    }
    if covered(state, place) {
        return;
    }
    state.retain(|value| !place.contains(value));
    state.insert(place.clone());
    let mut parent = place.clone();
    while parent.projection.pop().is_some() {
        let ty = place_type(module, function, &parent, &function.source).unwrap();
        let count = module.types.field_count(ty);
        let initialized = state
            .iter()
            .filter(|p| parent.contains(p) && p.projection.len() == parent.projection.len() + 1)
            .count();
        if initialized != count {
            break;
        }
        state.retain(|value| !parent.contains(value));
        state.insert(parent.clone());
    }
}

pub(crate) fn initialized(
    module: &Module,
    function: &Function,
    graph: &ControlFlowGraph,
) -> Result<(), Error> {
    let n = function.blocks.len();
    let all = (0..function.locals.len())
        .map(Place::local)
        .collect::<State>();
    let entry = (1..=function.parameters)
        .map(Place::local)
        .collect::<State>();
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
                let mut value = outputs[graph.predecessors(id)[0]].clone();
                for &pred in &graph.predecessors(id)[1..] {
                    value = intersect(&value, &outputs[pred]);
                }
                value
            };
            let mut output = input.clone();
            for statement in &block.statements {
                mark(module, function, &mut output, &statement.destination);
            }
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
            check_indices(&initialized, &statement.destination, &statement.source)?;
            for operand in statement.value.operands() {
                check(function, &initialized, operand, &statement.source)?;
            }
            mark(module, function, &mut initialized, &statement.destination);
        }
        match &block.terminator {
            Terminator::Branch { condition, .. }
            | Terminator::Assert { condition, .. }
            | Terminator::Switch {
                discriminant: condition,
                ..
            } => check(function, &initialized, condition, &block.source)?,
            Terminator::Return
                if function.locals[0] != Type::Unit && !covered(&initialized, &Place::local(0)) =>
            {
                return Err(Error::new(&block.source, "uninitialized return value"));
            }
            _ => {}
        }
    }
    Ok(())
}

fn check(
    function: &Function,
    initialized: &State,
    operand: &Operand,
    source: &SourceLocation,
) -> Result<(), Error> {
    if let Operand::Place(place) = operand {
        check_indices(initialized, place, source)?;
        if function.locals[place.local] != Type::Unit && !covered(initialized, place) {
            return Err(Error::new(
                source,
                format!(
                    "read of uninitialized local {} projection {:?}",
                    place.local, place.projection
                ),
            ));
        }
    }
    Ok(())
}

fn check_indices(initialized: &State, place: &Place, source: &SourceLocation) -> Result<(), Error> {
    for projection in &place.projection {
        if let Projection::Index(index) = projection
            && !covered(initialized, &Place::local(*index))
        {
            return Err(Error::new(source, "read of uninitialized array index"));
        }
    }
    Ok(())
}
