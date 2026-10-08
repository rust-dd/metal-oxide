mod analysis;
mod checks;
mod domain;
mod predicate;
mod refine;
mod state;
mod transfer;

use crate::*;

pub(crate) fn prove_numerics(module: &Module) -> Result<(), Error> {
    let needed = module.functions.iter().any(|function| {
        function.blocks.iter().any(|block| {
            matches!(block.terminator, Terminator::Assert { enabled: true, .. })
                || control_operand(&block.terminator).is_some_and(dynamic_operand)
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
        return Ok(());
    }
    let kernels = module.functions.iter().any(|function| function.kernel);
    let mut analysis = analysis::Analysis {
        module,
        remaining: 100_000,
        results: Default::default(),
    };
    for (index, function) in module.functions.iter().enumerate() {
        if function.kernel || !kernels {
            analysis.function(index, &vec![state::Value::default(); function.parameters])?;
        }
    }
    Ok(())
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

fn control_operand(terminator: &Terminator) -> Option<&Operand> {
    match terminator {
        Terminator::Branch { condition, .. } | Terminator::Assert { condition, .. } => {
            Some(condition)
        }
        Terminator::Switch { discriminant, .. } => Some(discriminant),
        _ => None,
    }
}
