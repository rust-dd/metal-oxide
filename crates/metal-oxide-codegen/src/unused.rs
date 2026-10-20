use std::collections::BTreeSet;

use metal_oxide_ir::{
    Expression, Function, FunctionAnalysis, Module, Operand, Place, Projection, Terminator, Type,
    operand_type, place_type,
};

/// Statements and locals needed by observable effects, returns, and unproved checks.
pub(crate) struct Used {
    statements: Vec<Vec<bool>>,
    locals: Vec<bool>,
}

impl Used {
    pub(crate) fn new(module: &Module, function: &Function, analysis: &FunctionAnalysis) -> Self {
        let aliases = Aliases::new(module, function, analysis);
        let cfg = &analysis.cfg;
        let mut statements = function
            .blocks
            .iter()
            .map(|block| vec![false; block.statements.len()])
            .collect::<Vec<_>>();
        let mut inputs = vec![vec![false; function.locals.len()]; function.blocks.len()];
        let mut pending = cfg.reachable_blocks().collect::<Vec<_>>();
        let mut queued = vec![true; function.blocks.len()];
        while let Some(id) = pending.pop() {
            queued[id] = false;
            let block = &function.blocks[id];
            let mut live = vec![false; function.locals.len()];
            for &successor in cfg.successors(id) {
                for (local, &needed) in inputs[successor].iter().enumerate() {
                    live[local] |= needed;
                }
            }
            terminator_reads(
                function,
                &block.terminator,
                analysis.checks.contains(&id),
                &mut |p| {
                    aliases.read_place(p, false, &mut |local| live[local] = true);
                },
            );
            for (index, statement) in block.statements.iter().enumerate().rev() {
                let destination = &statement.destination;
                let keep = destination.is_indirect()
                    || !removable(&statement.value)
                    || live[destination.local];
                statements[id][index] = keep;
                if !keep {
                    continue;
                }
                if destination.projection.is_empty() {
                    live[destination.local] = false;
                }
                let mut read = |local| live[local] = true;
                aliases.read_place(destination, true, &mut read);
                if let Expression::Ref { place, .. } = &statement.value {
                    operand_places(place, &mut |place| {
                        aliases.read_place(place, true, &mut read);
                    });
                } else {
                    for operand in statement.value.operands() {
                        operand_places(operand, &mut |place| {
                            aliases.read_place(place, false, &mut read);
                        });
                        if !removable(&statement.value) {
                            aliases.read_referents(module, function, operand, &mut read);
                        }
                    }
                }
            }
            if inputs[id] != live {
                inputs[id] = live;
                for &predecessor in cfg.predecessors(id) {
                    if !queued[predecessor] {
                        queued[predecessor] = true;
                        pending.push(predecessor);
                    }
                }
            }
        }
        let mut locals = vec![false; function.locals.len()];
        for id in cfg.reachable_blocks() {
            let block = &function.blocks[id];
            let mut declare = |place: &Place| {
                locals[place.local] = true;
                place_indices(place, &mut |local| locals[local] = true);
            };
            for (index, statement) in block.statements.iter().enumerate() {
                if statements[id][index] {
                    declare(&statement.destination);
                    for operand in statement.value.operands() {
                        operand_places(operand, &mut declare);
                    }
                }
            }
            terminator_reads(
                function,
                &block.terminator,
                analysis.checks.contains(&id),
                &mut declare,
            );
        }
        Self { statements, locals }
    }

    pub(crate) fn statement(&self, block: usize, index: usize) -> bool {
        self.statements[block][index]
    }

    pub(crate) fn local(&self, local: usize) -> bool {
        self.locals[local]
    }
}

#[derive(Clone, Default)]
struct Origins {
    locals: BTreeSet<usize>,
    unknown: bool,
}

impl Origins {
    fn extend(&mut self, other: &Self) -> bool {
        let old = (self.locals.len(), self.unknown);
        self.locals.extend(&other.locals);
        self.unknown |= other.unknown;
        old != (self.locals.len(), self.unknown)
    }
}

/// Flow-insensitive pointer origins; liveness of the referenced storage remains flow-sensitive.
struct Aliases {
    origins: Vec<Origins>,
    address_taken: BTreeSet<usize>,
}

impl Aliases {
    fn new(module: &Module, function: &Function, analysis: &FunctionAnalysis) -> Self {
        let mut result = Self {
            origins: vec![Origins::default(); function.locals.len()],
            address_taken: BTreeSet::new(),
        };
        let mut edges = vec![BTreeSet::new(); function.locals.len()];
        let mut may_mutate = false;
        for local in 1..=function.parameters {
            result.origins[local].unknown = contains_pointer(module, function.locals[local]);
        }
        for id in analysis.cfg.reachable_blocks() {
            for statement in &function.blocks[id].statements {
                let destination = &statement.destination;
                may_mutate |=
                    destination.is_indirect() || matches!(statement.value, Expression::Call { .. });
                if let Expression::Ref {
                    place: Operand::Place(place),
                    ..
                } = &statement.value
                    && !place.is_indirect()
                {
                    result.address_taken.insert(place.local);
                }
                let ty = place_type(module, function, destination, &statement.source).unwrap();
                if destination.is_indirect() || !contains_pointer(module, ty) {
                    continue;
                }
                let local = destination.local;
                match &statement.value {
                    Expression::Ref {
                        place: Operand::Place(place),
                        ..
                    } => {
                        let derefs = place
                            .projection
                            .iter()
                            .filter(|projection| matches!(projection, Projection::Deref))
                            .count();
                        match derefs {
                            0 => {
                                result.origins[local].locals.insert(place.local);
                            }
                            1 => {
                                edges[place.local].insert(local);
                            }
                            _ => result.origins[local].unknown = true,
                        }
                    }
                    Expression::Call { .. } | Expression::Uninit(_) => {
                        result.origins[local].unknown = true;
                    }
                    _ => {
                        for operand in statement.value.operands() {
                            let ty =
                                operand_type(module, function, operand, &statement.source).unwrap();
                            if contains_pointer(module, ty) {
                                operand_places(operand, &mut |place| {
                                    if place.is_indirect() {
                                        result.origins[local].unknown = true;
                                    } else {
                                        edges[place.local].insert(local);
                                    }
                                });
                            }
                        }
                    }
                }
            }
        }
        // Indirect writes and callees can change references stored inside borrowed aggregates.
        if may_mutate {
            for &local in &result.address_taken {
                result.origins[local].unknown |= contains_pointer(module, function.locals[local]);
            }
        }
        let mut pending = (0..function.locals.len()).collect::<Vec<_>>();
        let mut queued = vec![true; function.locals.len()];
        while let Some(local) = pending.pop() {
            queued[local] = false;
            let origins = result.origins[local].clone();
            for &destination in &edges[local] {
                if result.origins[destination].extend(&origins) && !queued[destination] {
                    queued[destination] = true;
                    pending.push(destination);
                }
            }
        }
        result
    }

    fn targets(&self, local: usize) -> impl Iterator<Item = usize> + '_ {
        let origins = &self.origins[local];
        origins.locals.iter().copied().chain(
            self.address_taken
                .iter()
                .copied()
                .filter(|_| origins.unknown),
        )
    }

    fn read_place(&self, place: &Place, address_only: bool, visit: &mut dyn FnMut(usize)) {
        place_indices(place, visit);
        let derefs = place
            .projection
            .iter()
            .filter(|projection| matches!(projection, Projection::Deref))
            .count();
        if derefs == 0 {
            if !address_only {
                visit(place.local);
            }
            return;
        }
        visit(place.local);
        let mut roots = BTreeSet::from([place.local]);
        for depth in 0..derefs {
            roots = roots
                .iter()
                .flat_map(|&local| self.targets(local))
                .collect();
            if !address_only || depth + 1 < derefs {
                for &local in &roots {
                    visit(local);
                }
            }
        }
    }

    fn read_referents(
        &self,
        module: &Module,
        function: &Function,
        operand: &Operand,
        visit: &mut dyn FnMut(usize),
    ) {
        let ty = operand_type(module, function, operand, &function.source).unwrap();
        if !contains_pointer(module, ty) {
            return;
        }
        let mut pending = Vec::new();
        operand_places(operand, &mut |place| {
            if place.is_indirect() {
                pending.extend(&self.address_taken);
            } else {
                pending.extend(self.targets(place.local));
            }
        });
        let mut seen = BTreeSet::new();
        while let Some(local) = pending.pop() {
            if seen.insert(local) {
                visit(local);
                pending.extend(self.targets(local));
            }
        }
    }
}

fn contains_pointer(module: &Module, ty: Type) -> bool {
    match ty {
        Type::Pointer(_) => true,
        Type::Aggregate(id) => module
            .types
            .get(id)
            .unwrap()
            .component_types()
            .any(|ty| contains_pointer(module, ty)),
        _ => false,
    }
}

fn removable(value: &Expression) -> bool {
    !matches!(
        value,
        Expression::Call { .. }
            | Expression::Log { .. }
            | Expression::BufferStore { .. }
            | Expression::Atomic { .. }
            | Expression::Simd { .. }
            | Expression::Matrix { .. }
            | Expression::ThreadgroupAlloc { .. }
            | Expression::ThreadgroupBarrier
    )
}

fn terminator_reads(
    function: &Function,
    terminator: &Terminator,
    runtime_check: bool,
    visit: &mut dyn FnMut(&Place),
) {
    match terminator {
        Terminator::Assert { .. } if !runtime_check => {}
        Terminator::Return if function.locals[0] != Type::Unit => visit(&Place::local(0)),
        _ => terminator
            .operands()
            .into_iter()
            .for_each(|operand| operand_places(operand, visit)),
    }
}

fn operand_places(operand: &Operand, visit: &mut dyn FnMut(&Place)) {
    match operand {
        Operand::Place(place) => visit(place),
        Operand::AggregateConstant { fields, .. } => {
            for field in fields {
                operand_places(field, visit);
            }
        }
        Operand::Constant(_) => {}
    }
}

fn place_indices(place: &Place, visit: &mut dyn FnMut(usize)) {
    for projection in &place.projection {
        if let Projection::Index(index) = projection {
            visit(*index);
        }
    }
}

#[cfg(test)]
mod tests;
