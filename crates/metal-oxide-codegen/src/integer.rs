use metal_oxide_ir::{IntegerOp, Scalar};

use crate::expressions::scalar_name;

/// MSL for an integer operation on `scalar` operands. Counts return `uint` like Rust's u32,
/// and rotation amounts are reduced modulo the bit width as Rust does.
pub(crate) fn integer(op: IntegerOp, scalar: Scalar, values: Vec<String>) -> String {
    let name = scalar_name(scalar);
    let mask = scalar.bits() - 1;
    match op {
        IntegerOp::Min => format!("min({}, {})", values[0], values[1]),
        IntegerOp::Max => format!("max({}, {})", values[0], values[1]),
        IntegerOp::SaturatingAdd => format!("addsat({}, {})", values[0], values[1]),
        IntegerOp::SaturatingSub => format!("subsat({}, {})", values[0], values[1]),
        IntegerOp::LeadingZeros => format!("uint(clz({}))", values[0]),
        IntegerOp::TrailingZeros => format!("uint(ctz({}))", values[0]),
        IntegerOp::CountOnes => format!("uint(popcount({}))", values[0]),
        IntegerOp::ReverseBits => format!("reverse_bits({})", values[0]),
        IntegerOp::RotateLeft => {
            format!("rotate({}, {name}({} & {mask}u))", values[0], values[1])
        }
        IntegerOp::RotateRight => {
            format!(
                "rotate({}, {name}((0u - {}) & {mask}u))",
                values[0], values[1]
            )
        }
    }
}
