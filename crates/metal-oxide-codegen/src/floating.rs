use metal_oxide_artifact::MathMode;
use metal_oxide_ir::{Expression, MathOp, Module, Scalar, sin_cos_type};

use crate::{expressions::scalar_name, scan::uses};

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
    } else if scalar == Scalar::Isize {
        "as_type<long>(0x8000000000000000ul)".into()
    } else {
        format!("{name}({minimum})")
    };
    let maximum = format!(
        "{name}({maximum}{})",
        if scalar.bits() == 64 {
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

pub(crate) fn math(op: MathOp, values: Vec<String>, math_mode: MathMode) -> String {
    match op {
        MathOp::Abs => {
            return format!("as_type<float>(as_type<uint>({}) & 0x7fffffffu)", values[0]);
        }
        MathOp::SinCos => return format!("metal_oxide_sin_cos({})", values[0]),
        _ => {}
    }
    let name = match op {
        MathOp::Min => "fmin",
        MathOp::Max => "fmax",
        MathOp::Sqrt => "sqrt",
        MathOp::Fma => "fma",
        MathOp::Ln => "log",
        MathOp::Exp => "exp",
        MathOp::Sin => "sin",
        MathOp::Cos => "cos",
        MathOp::Powf => "pow",
        MathOp::Tanh => "tanh",
        MathOp::Atan => "atan",
        MathOp::Floor => "floor",
        MathOp::Round => "round",
        MathOp::Ceil => "ceil",
        MathOp::Trunc => "trunc",
        MathOp::Abs | MathOp::SinCos => unreachable!(),
    };
    let namespace = if matches!(
        op,
        MathOp::Min
            | MathOp::Max
            | MathOp::Fma
            | MathOp::Floor
            | MathOp::Round
            | MathOp::Ceil
            | MathOp::Trunc
    ) {
        ""
    } else {
        match math_mode {
            MathMode::Precise => "precise::",
            MathMode::Fast => "fast::",
        }
    };
    let result = format!("{namespace}{name}({})", values.join(", "));
    if matches!(op, MathOp::Atan | MathOp::Round) && math_mode == MathMode::Precise {
        // Metal's precise atan and round can discard negative zero, including flushed inputs.
        format!("copysign({result}, {})", values[0])
    } else {
        result
    }
}

/// Defines `metal_oxide_sin_cos` when a function uses `MathOp::SinCos`.
pub(crate) fn helpers(module: &Module, math_mode: MathMode) -> String {
    let used = uses(module, |value| {
        matches!(
            value,
            Expression::Math {
                op: MathOp::SinCos,
                ..
            }
        )
    });
    if !used {
        return String::new();
    }
    let ty = crate::expressions::type_name(sin_cos_type(&module.types).expect("validated sin_cos"));
    let namespace = match math_mode {
        MathMode::Precise => "precise",
        MathMode::Fast => "fast",
    };
    format!(
        "static inline {ty} metal_oxide_sin_cos(float value) {{\n    float cosine;\n    float sine = {namespace}::sincos(value, cosine);\n    return {{sine, cosine}};\n}}\n\n"
    )
}
