use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::{
    domain::integer,
    state::{State, Value},
};
use crate::*;

pub(super) struct Analysis<'a> {
    pub module: &'a Module,
    pub remaining: usize,
    pub results: BTreeMap<(usize, Vec<Value>), Value>,
}

impl<'a> Analysis<'a> {
    pub fn function(&mut self, index: usize, arguments: &[Value]) -> Result<Value, Error> {
        let key = (index, arguments.to_vec());
        if let Some(result) = self.results.get(&key) {
            return Ok(result.clone());
        }
        let function = &self.module.functions[index];
        let graph = ControlFlowGraph::new(function)?;
        let dominators = graph.dominators();
        let loop_headers = (0..function.blocks.len())
            .map(|id| {
                graph
                    .predecessors(id)
                    .iter()
                    .any(|&from| dominators.dominates(id, from))
            })
            .collect::<Vec<_>>();
        let thresholds = thresholds(function);
        let mut entry = State::new(self.module, function);
        for (index, argument) in arguments.iter().enumerate() {
            entry.assign(&Place::local(index + 1), argument.clone());
        }
        let mut inputs = vec![Vec::<State<'a>>::new(); function.blocks.len()];
        let mut merged = vec![false; inputs.len()];
        let mut pending = VecDeque::from([(0, entry)]);
        let mut result: Option<Value> = None;
        while let Some((id, mut state)) = pending.pop_front() {
            if inputs[id].contains(&state) {
                continue;
            }
            if merged[id] {
                // Widen only at loop headers. Widening after a guard can discard
                // its bound even when every incoming path satisfies that guard.
                let joined = inputs[id][0].merge(&state, loop_headers[id].then_some(&thresholds));
                if joined == inputs[id][0] {
                    continue;
                }
                inputs[id][0] = joined.clone();
                state = joined;
            } else if inputs[id].len() >= 64 {
                for input in &inputs[id] {
                    state = state.merge(input, None);
                }
                inputs[id] = vec![state.clone()];
                merged[id] = true;
            } else {
                inputs[id].push(state.clone());
            }
            if self.remaining == 0 {
                return Err(Error::new(
                    &function.blocks[id].source,
                    "numerical proof analysis budget exceeded",
                ));
            }
            self.remaining -= 1;
            let block = &function.blocks[id];
            for statement in &block.statements {
                self.statement(&mut state, statement)?;
            }
            if let Some(Operand::Place(place)) = super::control_operand(&block.terminator) {
                super::checks::place(&state, place, &block.source)?;
            }
            match &block.terminator {
                Terminator::Goto(target) => pending.push_back((*target, state)),
                Terminator::Assert {
                    condition,
                    expected,
                    enabled,
                    target,
                    message,
                } => {
                    if *enabled && state.condition(condition) != Some(*expected) {
                        return Err(Error::new(
                            &block.source,
                            format!("MIR assertion cannot be proved safe: {message}"),
                        ));
                    }
                    pending.push_back((*target, state));
                }
                Terminator::Branch {
                    condition,
                    then_block,
                    else_block,
                } => {
                    let mut other = state.clone();
                    if state.assume(condition, true) {
                        pending.push_back((*then_block, state));
                    }
                    if other.assume(condition, false) {
                        pending.push_back((*else_block, other));
                    }
                }
                Terminator::Switch {
                    discriminant,
                    cases,
                    otherwise,
                } => {
                    for (constant, target) in cases {
                        let mut branch = state.clone();
                        if select(&mut branch, discriminant, integer(*constant).unwrap(), true) {
                            pending.push_back((*target, branch));
                        }
                    }
                    if cases
                        .iter()
                        .all(|(c, _)| select(&mut state, discriminant, integer(*c).unwrap(), false))
                    {
                        pending.push_back((*otherwise, state));
                    }
                }
                Terminator::Return => {
                    let value = state.operand_value(&Operand::local(0));
                    result = Some(result.map_or(value.clone(), |previous| previous.join(&value)));
                }
                Terminator::Unreachable => {}
            }
        }
        let result = result.unwrap_or_default();
        self.results.insert(key, result.clone());
        Ok(result)
    }
}

/// Integer constants and their neighbours bound loop widening to the program's own limits.
fn thresholds(function: &Function) -> BTreeSet<i128> {
    fn constants(operand: &Operand, values: &mut BTreeSet<i128>) {
        match operand {
            Operand::Constant(constant) => values.extend(integer(*constant)),
            Operand::AggregateConstant { fields, .. } => {
                fields.iter().for_each(|field| constants(field, values));
            }
            Operand::Place(_) => {}
        }
    }
    let mut values = BTreeSet::new();
    for block in &function.blocks {
        for statement in &block.statements {
            statement
                .value
                .operands()
                .into_iter()
                .for_each(|operand| constants(operand, &mut values));
        }
        if let Terminator::Switch { cases, .. } = &block.terminator {
            values.extend(cases.iter().filter_map(|(constant, _)| integer(*constant)));
        }
    }
    values
        .iter()
        .flat_map(|&value| [value - 1, value, value + 1])
        .collect()
}

fn select(state: &mut State<'_>, operand: &Operand, value: i128, equal: bool) -> bool {
    if let Some(constant) = state.constant(operand) {
        return (constant == value) == equal;
    }
    if let Operand::Place(place) = operand {
        return state.restrict_place(place, value, equal);
    }
    true
}
