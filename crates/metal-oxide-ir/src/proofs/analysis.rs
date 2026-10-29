use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::{
    Aliasing,
    domain::integer,
    state::{State, Value},
};
use crate::*;

/// A function analyzed under its kernel's required block and its argument facts.
type Context = (usize, Option<[u32; 3]>, Vec<Value>);

pub(super) struct Analysis<'a> {
    module: &'a Module,
    aliasing: &'a [Aliasing],
    pub remaining: usize,
    /// The required block of the kernel being analyzed, which bounds launch coordinates.
    pub block: Option<[u32; 3]>,
    /// Runtime checks per function, joined over every analyzed calling context.
    pub checks: Vec<BTreeSet<usize>>,
    results: BTreeMap<Context, Value>,
    loop_headers: Vec<Vec<bool>>,
    joins: Vec<Vec<bool>>,
    thresholds: Vec<BTreeSet<i128>>,
}

impl<'a> Analysis<'a> {
    pub fn new(module: &'a Module, graphs: &[FunctionAnalysis], aliasing: &'a [Aliasing]) -> Self {
        let loop_headers = module
            .functions
            .iter()
            .zip(graphs)
            .map(|(function, graph)| loop_headers(function, graph))
            .collect::<Vec<_>>();
        Self {
            module,
            aliasing,
            remaining: 0,
            block: None,
            checks: vec![BTreeSet::new(); module.functions.len()],
            results: BTreeMap::new(),
            joins: module
                .functions
                .iter()
                .zip(&loop_headers)
                .map(|(function, headers)| joins(function, headers))
                .collect(),
            loop_headers,
            thresholds: module.functions.iter().map(thresholds).collect(),
        }
    }

    pub fn function(&mut self, index: usize, arguments: &[Value]) -> Result<Value, Error> {
        let key = (index, self.block, arguments.to_vec());
        if let Some(result) = self.results.get(&key) {
            return Ok(result.clone());
        }
        let function = &self.module.functions[index];
        let mut entry = State::new(self.module, function, &self.aliasing[index]);
        for (position, argument) in arguments.iter().enumerate() {
            entry.assign(&Place::local(position + 1), argument.clone());
        }
        for contract in &function.requires {
            entry.require(contract);
        }
        let mut inputs = vec![Vec::<State<'a>>::new(); function.blocks.len()];
        let mut merged = vec![false; inputs.len()];
        let mut pending = VecDeque::from([(0, entry)]);
        let mut result = None::<Value>;
        while let Some((id, mut state)) = pending.pop_front() {
            if inputs[id].contains(&state) {
                continue;
            }
            if merged[id] {
                // Widen only at loop headers. Widening after a guard can discard
                // its bound even when every incoming path satisfies that guard.
                let widen = self.loop_headers[index][id].then_some(&self.thresholds[index]);
                if !inputs[id][0].merge(&state, widen) {
                    continue;
                }
                state = inputs[id][0].clone();
            } else if inputs[id].len() >= 64 && self.joins[index][id] {
                for input in &inputs[id] {
                    state.merge(input, None);
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
            for operand in block.terminator.operands() {
                if let Operand::Place(place) = operand {
                    super::checks::place(&state, place, &block.source)?;
                }
            }
            match &block.terminator {
                Terminator::Goto(target) => pending.push_back((*target, state)),
                Terminator::Assert {
                    condition,
                    expected,
                    enabled,
                    target,
                    ..
                } => {
                    if !*enabled {
                        pending.push_back((*target, state));
                    } else {
                        if state.condition(condition) != Some(*expected)
                            && !state.implied(condition, *expected)
                        {
                            self.checks[index].insert(id);
                        }
                        // Proven conditions are assumed too: slice indices rely on their truth.
                        if state.assume(condition, *expected) {
                            pending.push_back((*target, state));
                        }
                    }
                }
                Terminator::Panic(_) => {
                    self.checks[index].insert(id);
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

fn loop_headers(function: &Function, graph: &FunctionAnalysis) -> Vec<bool> {
    (0..function.blocks.len())
        .map(|id| {
            graph
                .cfg
                .predecessors(id)
                .iter()
                .any(|&from| graph.dominators.dominates(id, from))
        })
        .collect()
}

/// Loop headers and blocks with several incoming edges. Only these merge their states, so that
/// no merge separates a check from the operation that relies on it.
fn joins(function: &Function, loop_headers: &[bool]) -> Vec<bool> {
    let mut edges = vec![0_usize; function.blocks.len()];
    for block in &function.blocks {
        // A switch sends a state along each case, also when cases share a target.
        let targets = match &block.terminator {
            Terminator::Switch {
                cases, otherwise, ..
            } => cases
                .iter()
                .map(|(_, target)| *target)
                .chain([*otherwise])
                .collect(),
            terminator => terminator.successors(),
        };
        for target in targets {
            edges[target] += 1;
        }
    }
    edges
        .iter()
        .zip(loop_headers)
        .map(|(&edges, &header)| header || edges > 1)
        .collect()
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
