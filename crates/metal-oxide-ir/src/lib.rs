//! Typed, source-located compute kernel IR.

mod control_flow;
mod dataflow;
mod model;
mod typing;
mod uniform;
mod validate;

pub use control_flow::{ControlFlowGraph, Dominators};
pub use model::*;
pub use typing::operand_type;
pub use validate::validate;
