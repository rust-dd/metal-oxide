//! Typed, source-located compute kernel IR.

mod analysis;
mod control_flow;
mod dataflow;
mod model;
mod proofs;
mod types;
mod typing;
mod uniform;
mod validate;

pub use analysis::FunctionAnalysis;
pub use control_flow::{ControlFlowGraph, Dominators};
pub use model::*;
pub use proofs::prove_numerics;
pub use types::{Access, AddressSpace, Aggregate, Element, RecordField, Scalar, Type, TypeTable};
pub use typing::{operand_type, place_type};
pub use validate::validate;
