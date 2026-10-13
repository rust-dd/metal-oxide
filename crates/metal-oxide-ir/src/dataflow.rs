use std::collections::BTreeSet;

use crate::*;

type State = BTreeSet<Place>;

/// Places of one local; only these can contain or be contained by its places.
fn same_local(state: &State, local: usize) -> impl Iterator<Item = &Place> {
    state
        .range(Place::local(local)..)
        .take_while(move |place| place.local == local)
}

fn intersect(a: &State, b: &State) -> State {
    let mut result = State::new();
    for left in a {
        for right in same_local(b, left.local) {
            if left.contains(right) {
                result.insert(right.clone());
            } else if right.contains(left) {
                result.insert(left.clone());
            }
        }
    }
    result
}

fn remove_contained(state: &mut State, place: &Place) {
    let contained = same_local(state, place.local)
        .filter(|value| place.contains(value))
        .cloned()
        .collect::<Vec<_>>();
    for value in contained {
        state.remove(&value);
    }
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
    same_local(state, required.local).any(|value| value.contains(&required))
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
    remove_contained(state, place);
    state.insert(place.clone());
    let mut parent = place.clone();
    while parent.projection.pop().is_some() {
        let ty = place_type(module, function, &parent, &function.source).unwrap();
        let count = module.types.field_count(ty);
        let initialized = same_local(state, parent.local)
            .filter(|p| parent.contains(p) && p.projection.len() == parent.projection.len() + 1)
            .count();
        if initialized != count {
            break;
        }
        remove_contained(state, &parent);
        state.insert(parent.clone());
    }
}

pub(crate) fn initialized(
    module: &Module,
    function: &Function,
    graph: &ControlFlowGraph,
) -> Result<(), Error> {
    let n = function.blocks.len();
    let entry = (1..=function.parameters)
        .map(Place::local)
        .collect::<State>();
    // `None` stands for every local, the identity of `intersect`, until a block is first visited.
    let mut inputs = vec![None::<State>; n];
    let mut outputs = vec![None::<State>; n];
    loop {
        let mut changed = false;
        for (id, block) in function.blocks.iter().enumerate() {
            if !graph.is_reachable(id) {
                continue;
            }
            let input = if id == 0 {
                Some(entry.clone())
            } else {
                graph.predecessors(id).iter().fold(None, |value, &pred| {
                    match (value, &outputs[pred]) {
                        (value, None) => value,
                        (None, Some(output)) => Some(output.clone()),
                        (Some(value), Some(output)) => Some(intersect(&value, output)),
                    }
                })
            };
            let mut output = input.clone();
            if let Some(output) = &mut output {
                for statement in &block.statements {
                    mark(module, function, output, &statement.destination);
                }
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
        // Every reachable block has a path from the entry, so its input is known.
        let mut initialized = inputs[id].clone().unwrap();
        for statement in &block.statements {
            check_indices(&initialized, &statement.destination, &statement.source)?;
            for operand in statement.value.operands() {
                check(function, &initialized, operand, &statement.source)?;
            }
            mark(module, function, &mut initialized, &statement.destination);
        }
        for operand in block.terminator.operands() {
            check(function, &initialized, operand, &block.source)?;
        }
        if block.terminator == Terminator::Return
            && function.locals[0] != Type::Unit
            && !covered(&initialized, &Place::local(0))
        {
            return Err(Error::new(&block.source, "uninitialized return value"));
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
