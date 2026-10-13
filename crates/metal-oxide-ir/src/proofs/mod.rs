mod analysis;
mod checks;
mod domain;
mod predicate;
mod refine;
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
    let mut analysis = analysis::Analysis::new(module, graphs);
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

fn dynamic_operand(operand: &Operand) -> bool {
    matches!(operand, Operand::Place(place) if dynamic(place))
}

fn dynamic(place: &Place) -> bool {
    place
        .projection
        .iter()
        .any(|p| matches!(p, Projection::Index(_)))
}
