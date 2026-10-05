//! Typed, source-located compute kernel IR.

mod dataflow;
mod model;
mod typing;
mod uniform;
mod validate;

pub use model::*;
pub use typing::operand_type;
pub use validate::validate;
