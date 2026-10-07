use metal_oxide_ir::{MathOp, Scalar};

use crate::expressions::scalar_name;

pub(crate) fn saturating_cast(value: &str, scalar: Scalar) -> String {
    let (minimum, maximum) = if scalar.is_signed() {
        let limit = 1_i128 << (scalar.bits() - 1);
        (-limit, limit - 1)
    } else {
        (0, (1_i128 << scalar.bits()) - 1)
    };
    let lower = (minimum as f32).to_bits();
    let upper = (maximum as f32).to_bits();
    let name = scalar_name(scalar);
    let minimum = if scalar == Scalar::I32 {
        "as_type<int>(0x80000000u)".into()
    } else {
        format!("{name}({minimum})")
    };
    let maximum = format!(
        "{name}({maximum}{})",
        if scalar == Scalar::Usize {
            "ul"
        } else if scalar.is_signed() {
            ""
        } else {
            "u"
        }
    );
    format!(
        "(isnan({value}) ? {name}(0) : ({value} <= as_type<float>(0x{lower:08x}u) ? {minimum} : ({value} >= as_type<float>(0x{upper:08x}u) ? {maximum} : {name}({value}))))"
    )
}

pub(crate) fn math(op: MathOp, values: Vec<String>) -> String {
    if op == MathOp::Abs {
        return format!("as_type<float>(as_type<uint>({}) & 0x7fffffffu)", values[0]);
    }
    let name = match op {
        MathOp::Min => "fmin",
        MathOp::Max => "fmax",
        MathOp::Sqrt => "precise::sqrt",
        MathOp::Fma => "fma",
        MathOp::Abs => unreachable!(),
    };
    format!("{name}({})", values.join(", "))
}
