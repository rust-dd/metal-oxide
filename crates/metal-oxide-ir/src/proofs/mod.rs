mod analysis;
mod checks;
mod domain;
mod predicate;
mod refine;
mod relations;
mod state;
mod transfer;

use std::collections::BTreeSet;

use crate::*;

/// Proves numerical checks; returns per function the blocks whose unproved checks run on the
/// GPU. Their success edges assume the checked condition.
pub(crate) fn prove_numerics(
    module: &Module,
    graphs: &[FunctionAnalysis],
) -> Result<Vec<BTreeSet<usize>>, Error> {
    let needed = module.functions.iter().any(|function| {
        function.blocks.iter().any(|block| {
            matches!(
                block.terminator,
                Terminator::Assert { enabled: true, .. } | Terminator::Panic(_)
            ) || block.terminator.operands().into_iter().any(dynamic_operand)
                || block.statements.iter().any(|statement| {
                    let division = match &statement.value {
                        Expression::Binary(BinaryOp::Div | BinaryOp::Rem, value, _) => {
                            matches!(
                                operand_type(module, function, value, &statement.source),
                                Ok(Type::Scalar(scalar)) if scalar.is_integer()
                            )
                        }
                        _ => false,
                    };
                    division
                        || dynamic(&statement.destination)
                        || statement.value.operands().into_iter().any(dynamic_operand)
                })
        })
    });
    if !needed {
        return Ok(vec![BTreeSet::new(); module.functions.len()]);
    }
    let kernels = module.functions.iter().any(|function| function.kernel);
    let aliasing = module
        .functions
        .iter()
        .map(|function| Aliasing::new(module, function))
        .collect::<Vec<_>>();
    let mut analysis = analysis::Analysis::new(module, graphs, &aliasing);
    for (index, function) in module.functions.iter().enumerate() {
        if function.kernel || !kernels {
            // Completed summaries are keyed by function and argument facts;
            // sharing them is safe, but unrelated roots must not share a budget.
            analysis.remaining = 100_000;
            analysis.block = function.required_block;
            analysis.function(index, &vec![state::Value::default(); function.parameters])?;
        }
    }
    Ok(analysis.checks)
}

/// Which places writes through pointers and calls may change.
#[derive(Debug)]
pub(super) struct Aliasing {
    /// Locals borrowed mutably, which writes through pointers and calls may change.
    pub escaped: Vec<usize>,
    /// Mutable reference parameters only read and written through: no other pointer reaches
    /// their referent while the function runs, so it resolves like a shared referent.
    pub exclusive: Vec<usize>,
}

impl Aliasing {
    fn new(module: &Module, function: &Function) -> Self {
        let statements = function.blocks.iter().flat_map(|block| &block.statements);
        let mut escaped = statements
            .clone()
            .filter_map(|statement| match &statement.value {
                Expression::Ref {
                    place: Operand::Place(place),
                    mutable: true,
                } if !place.is_indirect() => Some(place.local),
                _ => None,
            })
            .collect::<Vec<_>>();
        escaped.sort_unstable();
        escaped.dedup();
        let through = |place: &Place| place.projection.first() == Some(&Projection::Deref);
        let mut exclusive = (1..=function.parameters)
            .filter(|&local| {
                module
                    .types
                    .pointer(function.locals[local])
                    .is_some_and(|pointer| pointer.mutable && !pointer.slice)
            })
            .collect::<Vec<_>>();
        let mut used = |place: &Place, reborrowed: bool| {
            if !through(place) || reborrowed {
                exclusive.retain(|&local| local != place.local);
            }
        };
        for statement in statements {
            used(&statement.destination, false);
            for operand in statement.value.operands() {
                if let Operand::Place(place) = operand {
                    let reborrowed = matches!(
                        &statement.value,
                        Expression::Ref { place: Operand::Place(borrowed), mutable: true }
                            if borrowed == place
                    );
                    used(place, reborrowed);
                }
            }
        }
        for block in &function.blocks {
            for operand in block.terminator.operands() {
                if let Operand::Place(place) = operand {
                    used(place, false);
                }
            }
        }
        Self { escaped, exclusive }
    }
}

fn dynamic_operand(operand: &Operand) -> bool {
    matches!(operand, Operand::Place(place) if dynamic(place))
}

fn dynamic(place: &Place) -> bool {
    place
        .projection
        .iter()
        .any(|p| matches!(p, Projection::Index(_)))
}
