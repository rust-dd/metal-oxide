mod analysis;
mod checks;
mod domain;
mod predicate;
mod refine;
mod state;
mod transfer;

use crate::*;

/// Proves enabled assertions, integer division guards and dynamic array bounds.
///
/// Call after structural validation. Unknown facts and exhausted analysis budgets
/// are errors; this function does not modify or disable IR assertions.
pub fn prove_numerics(module: &Module) -> Result<(), Error> {
    let needed = module.functions.iter().any(|function| function.blocks.iter().any(|block| {
        matches!(block.terminator, Terminator::Assert { enabled: true, .. })
            || control_operand(&block.terminator).is_some_and(|operand| matches!(operand, Operand::Place(p) if dynamic(p)))
            || block.statements.iter().any(|statement| {
                matches!(&statement.value, Expression::Binary(BinaryOp::Div | BinaryOp::Rem, a, _) if matches!(operand_type(module, function, a, &statement.source), Ok(Type::Scalar(s)) if s.is_integer()))
                    || dynamic(&statement.destination)
                    || statement.value.operands().iter().any(|operand| matches!(operand, Operand::Place(p) if dynamic(p)))
            })
    }));
    if !needed {
        return Ok(());
    }
    let kernels = module.functions.iter().any(|function| function.kernel);
    let mut analysis = analysis::Analysis {
        module,
        remaining: 100_000,
    };
    for (index, function) in module.functions.iter().enumerate() {
        if function.kernel || !kernels {
            analysis.function(index, &vec![state::Value::default(); function.parameters])?;
        }
    }
    Ok(())
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
