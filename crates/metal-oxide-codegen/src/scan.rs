use metal_oxide_ir::{Expression, Module};

/// Whether some statement of the module evaluates an expression that `operation` accepts.
pub(crate) fn uses(module: &Module, operation: impl Fn(&Expression) -> bool) -> bool {
    module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.statements)
        .any(|statement| operation(&statement.value))
}
