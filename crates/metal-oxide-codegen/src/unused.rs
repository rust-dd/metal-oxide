use metal_oxide_ir::{
    ControlFlowGraph, Expression, Function, Operand, Place, Projection, Terminator, Type,
};

/// Statements and locals the emitted MSL needs. Validated assertions emit no code,
/// so values that only feed their conditions are dropped with them.
pub(crate) struct Used {
    statements: Vec<Vec<bool>>,
    locals: Vec<bool>,
}

impl Used {
    pub(crate) fn new(function: &Function, cfg: &ControlFlowGraph) -> Self {
        let mut statements = function
            .blocks
            .iter()
            .enumerate()
            .map(|(id, block)| vec![cfg.is_reachable(id); block.statements.len()])
            .collect::<Vec<_>>();
        let mut reads = vec![0_usize; function.locals.len()];
        let mut writers = vec![Vec::new(); function.locals.len()];
        for (id, block) in function.blocks.iter().enumerate() {
            if !cfg.is_reachable(id) {
                continue;
            }
            for (index, statement) in block.statements.iter().enumerate() {
                statement_reads(&statement.destination, &statement.value, &mut |local| {
                    reads[local] += 1
                });
                writers[statement.destination.local].push((id, index));
            }
            terminator_reads(function, &block.terminator, &mut |local| reads[local] += 1);
        }
        let mut pending = (0..reads.len())
            .filter(|&local| reads[local] == 0)
            .collect::<Vec<_>>();
        while let Some(local) = pending.pop() {
            for &(id, index) in &writers[local] {
                let statement = &function.blocks[id].statements[index];
                if !statements[id][index] || !removable(&statement.value) {
                    continue;
                }
                statements[id][index] = false;
                statement_reads(&statement.destination, &statement.value, &mut |read| {
                    reads[read] -= 1;
                    if reads[read] == 0 {
                        pending.push(read);
                    }
                });
            }
        }
        let mut locals = vec![false; function.locals.len()];
        for (id, block) in function.blocks.iter().enumerate() {
            if !cfg.is_reachable(id) {
                continue;
            }
            for (index, statement) in block.statements.iter().enumerate() {
                if statements[id][index] {
                    locals[statement.destination.local] = true;
                    statement_reads(&statement.destination, &statement.value, &mut |local| {
                        locals[local] = true
                    });
                }
            }
            terminator_reads(function, &block.terminator, &mut |local| {
                locals[local] = true
            });
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

/// Expressions without effects beyond their value; calls stay because helpers may store.
fn removable(value: &Expression) -> bool {
    !matches!(
        value,
        Expression::Call { .. }
            | Expression::BufferStore { .. }
            | Expression::Atomic { .. }
            | Expression::Simd { .. }
            | Expression::ThreadgroupAlloc { .. }
            | Expression::ThreadgroupBarrier
    )
}

fn statement_reads(destination: &Place, value: &Expression, visit: &mut impl FnMut(usize)) {
    place_indices(destination, visit);
    for operand in value.operands() {
        operand_reads(operand, visit);
    }
}

fn terminator_reads(function: &Function, terminator: &Terminator, visit: &mut impl FnMut(usize)) {
    match terminator {
        Terminator::Branch { condition, .. } => operand_reads(condition, visit),
        Terminator::Switch { discriminant, .. } => operand_reads(discriminant, visit),
        Terminator::Return if function.locals[0] != Type::Unit => visit(0),
        _ => {}
    }
}

fn operand_reads(operand: &Operand, visit: &mut impl FnMut(usize)) {
    match operand {
        Operand::Place(place) => {
            visit(place.local);
            place_indices(place, visit);
        }
        Operand::AggregateConstant { fields, .. } => {
            for field in fields {
                operand_reads(field, visit);
            }
        }
        Operand::Constant(_) => {}
    }
}

fn place_indices(place: &Place, visit: &mut impl FnMut(usize)) {
    for projection in &place.projection {
        if let Projection::Index(index) = projection {
            visit(*index);
        }
    }
}
