//! Typed, source-located compute kernel IR.

mod analysis;
mod atomic;
mod check;
mod control_flow;
mod dataflow;
mod log;
mod matrix;
mod model;
mod proofs;
mod simd;
mod types;
mod typing;
mod uniform;
mod validate;

pub use analysis::FunctionAnalysis;
pub use check::Check;
pub use control_flow::{ControlFlowGraph, Dominators};
pub use log::{LogPiece, log_pieces};
pub use matrix::{MatrixOp, is_matrix_element};
pub use metal_oxide_artifact::{Access, AtomicOp, AtomicOrdering, OverflowOp};
pub use model::*;
pub use simd::{SimdOp, ballot_type};
pub use types::{AddressSpace, Aggregate, Element, Pointer, RecordField, Scalar, Type, TypeTable};
pub use typing::{Located, PlaceInfo, expression_type, operand_type, place_info, place_type};
pub use validate::validate;
