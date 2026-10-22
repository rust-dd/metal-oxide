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
        IntegerOp::TrailingZeros if scalar.bits() < 32 => {
            format!(
                "ctz({} | {:#x}u)",
                widened(scalar, &values[0]),
                1_u32 << scalar.bits()
            )
        }
        IntegerOp::TrailingZeros => format!("uint(ctz({}))", values[0]),
        IntegerOp::CountOnes if scalar.bits() < 32 => {
            format!("popcount({})", widened(scalar, &values[0]))
        }
        IntegerOp::CountOnes => format!("uint(popcount({}))", values[0]),
        IntegerOp::SwapBytes => swap_bytes(scalar, &values[0]),
        IntegerOp::ReverseBits => {
            // Hosted Metal miscomputes float arithmetic fed by reversed constants.
            let value = crate::arithmetic::integer_value(
                scalar,
                &format!("({} ^ {})", values[0], crate::arithmetic::OPAQUE_ZERO),
            );
            format!("reverse_bits({value})")
        }
        IntegerOp::RotateLeft if scalar.bits() < 32 => {
            rotated(scalar, &values[0], &format!("({} & {mask}u)", values[1]))
        }
        IntegerOp::RotateRight if scalar.bits() < 32 => rotated(
            scalar,
            &values[0],
            &format!("((0u - {}) & {mask}u)", values[1]),
        ),
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

fn swap_bytes(scalar: Scalar, value: &str) -> String {
    if scalar.bits() == 8 {
        return value.to_owned();
    }
    let (wide, suffix) = if scalar.bits() == 64 {
        ("ulong", "ul")
    } else {
        ("uint", "u")
    };
    let bytes = scalar.bits() / 8;
    let parts = (0..bytes)
        .map(|index| {
            let from = index * 8;
            let to = (bytes - 1 - index) * 8;
            format!("((({wide}({value}) >> {from}u) & 0xff{suffix}) << {to}u)")
        })
        .collect::<Vec<_>>();
    crate::arithmetic::integer_value(scalar, &parts.join(" | "))
}

/// An 8- or 16-bit `value` rotated left by `amount`, below its width, as a 32-bit rotation of
/// its copies: hosted macOS 15 Metal rotates a narrow value with bits a truncation dropped.
fn rotated(scalar: Scalar, value: &str, amount: &str) -> String {
    let copies = if scalar.bits() == 8 {
        "0x1010101u"
    } else {
        "0x10001u"
    };
    let wide = widened(scalar, value);
    crate::arithmetic::integer_value(scalar, &format!("rotate({wide} * {copies}, {amount})"))
}

/// `value` zero-extended to `uint` and kept unknown: hosted Metal can fail on 8- and
/// 16-bit counts of values it folds, and narrow a plain extension back.
fn widened(scalar: Scalar, value: &str) -> String {
    let unsigned = if scalar.bits() == 8 {
        "uchar"
    } else {
        "ushort"
    };
    format!(
        "(uint(as_type<{unsigned}>({value})) ^ {})",
        crate::arithmetic::OPAQUE_ZERO
    )
}
