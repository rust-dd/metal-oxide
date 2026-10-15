use metal_oxide_ir::*;

use crate::expressions::scalar_name;

fn integer_value(scalar: Scalar, value: &str) -> String {
    let name = scalar_name(scalar);
    if scalar.is_signed() {
        let unsigned = match scalar.bits() {
            8 => "uchar",
            16 => "ushort",
            32 => "uint",
            64 => "ulong",
            _ => unreachable!(),
        };
        format!("as_type<{name}>({unsigned}({value}))")
    } else {
        format!("{name}({value})")
    }
}

pub(crate) fn cast(
    value: &str,
    from: Type,
    to: Scalar,
    source: &SourceLocation,
) -> Result<String, Error> {
    if from == Type::Scalar(Scalar::F32) && to.is_integer() {
        return Ok(crate::floating::saturating_cast(value, to));
    }
    if (to == Scalar::Bool && from != Type::Scalar(Scalar::Bool))
        || (from == Type::Scalar(Scalar::Bool) && to == Scalar::F32)
    {
        return Err(Error::new(source, "unsupported boolean cast"));
    }
    if let Some(helper) = wide_to_float(from, to) {
        return Ok(format!("{helper}({value})"));
    }
    Ok(if to.is_integer() {
        integer_value(to, value)
    } else {
        format!("{}({value})", scalar_name(to))
    })
}

/// The Apple Paravirtual device folds a 64-bit integer that becomes constant on the device into
/// a wrong float; these convert through 32 bits, rounding to nearest even like Rust.
pub(crate) const WIDE_TO_FLOAT: &str = "static inline float metal_oxide_u64_to_f32(ulong x) {
    uint high = uint(x >> 32);
    if (high == 0u) {
        return float(uint(x));
    }
    uint shift = 32u - clz(high);
    uint sticky = uint((x & ((1ul << shift) - 1ul)) != 0ul);
    return ldexp(float(uint(x >> shift) | sticky), int(shift));
}

static inline float metal_oxide_i64_to_f32(long x) {
    float magnitude = metal_oxide_u64_to_f32(x < 0 ? 0ul - ulong(x) : ulong(x));
    return x < 0 ? -magnitude : magnitude;
}

";

/// The [`WIDE_TO_FLOAT`] helper that converts `from` to `to`, if one does.
pub(crate) fn wide_to_float(from: Type, to: Scalar) -> Option<&'static str> {
    match (from, to) {
        (Type::Scalar(Scalar::U64 | Scalar::Usize), Scalar::F32) => Some("metal_oxide_u64_to_f32"),
        (Type::Scalar(Scalar::Isize), Scalar::F32) => Some("metal_oxide_i64_to_f32"),
        _ => None,
    }
}

pub(crate) fn unary(op: UnaryOp, value: &str, ty: Type) -> String {
    match (op, ty) {
        (UnaryOp::Neg, Type::Scalar(Scalar::Isize)) => {
            integer_value(Scalar::Isize, &format!("0ul - ulong({value})"))
        }
        (UnaryOp::Neg, Type::Scalar(scalar)) if scalar.is_signed() => {
            integer_value(scalar, &format!("0u - uint(int({value}))"))
        }
        (UnaryOp::Neg, _) => format!("(-{value})"),
        (UnaryOp::Not, Type::Scalar(Scalar::Bool)) => format!("(!{value})"),
        (UnaryOp::Not, Type::Scalar(Scalar::U64 | Scalar::Usize | Scalar::Isize)) => {
            format!("(~{value})")
        }
        (UnaryOp::Not, Type::Scalar(scalar)) => integer_value(scalar, &format!("~uint({value})")),
        _ => unreachable!("validated unary type"),
    }
}

/// Zero, as a thread's position in its block is below the block size, but unknown to Metal's
/// GPU compiler, which mishandles some values it folds: combined with it, they stay unknown.
pub(crate) const OPAQUE_ZERO: &str =
    "uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x)";

/// The numerator of a 32-bit division by a variable, which Metal misdivides when it is constant.
pub(crate) fn division_numerator(op: BinaryOp, a: String, b: &Operand, ty: Type) -> String {
    if !matches!(op, BinaryOp::Div | BinaryOp::Rem) || matches!(b, Operand::Constant(_)) {
        return a;
    }
    match ty {
        Type::Scalar(Scalar::U32) => format!("({a} ^ {OPAQUE_ZERO})"),
        Type::Scalar(Scalar::I32) => format!("({a} ^ as_type<int>({OPAQUE_ZERO}))"),
        _ => a,
    }
}

pub(crate) fn binary(op: BinaryOp, a: &str, b: &str, ty: Type) -> String {
    use BinaryOp::*;
    let symbol = match op {
        Add | AddWithOverflow => "+",
        Sub | SubWithOverflow => "-",
        Mul | MulWithOverflow => "*",
        Div => "/",
        Rem => "%",
        BitAnd => "&",
        BitOr => "|",
        BitXor => "^",
        Shl => "<<",
        Shr => ">>",
        Eq => "==",
        Ne => "!=",
        Lt => "<",
        Le => "<=",
        Gt => ">",
        Ge => ">=",
        Offset => "+",
        OffsetFrom => "-",
    };
    if matches!(ty, Type::Pointer(_)) {
        // Pointers compare, move by element counts and subtract to an element count.
        return match op {
            OffsetFrom => format!("ulong({a} - {b})"),
            _ => format!("({a} {symbol} {b})"),
        };
    }
    if matches!(op, AddWithOverflow | SubWithOverflow | MulWithOverflow) {
        let Type::Scalar(scalar) = ty else {
            unreachable!()
        };
        let label = match op {
            AddWithOverflow => "add",
            SubWithOverflow => "sub",
            _ => "mul",
        };
        return format!("metal_oxide_{label}_checked_{}({a}, {b})", scalar.name());
    }
    if ty == Type::Scalar(Scalar::F32) && op == Rem {
        return format!("fmod({a}, {b})");
    }
    let Type::Scalar(scalar) = ty else {
        unreachable!("validated binary type")
    };
    if scalar == Scalar::Isize && matches!(op, Add | Sub | Mul) {
        return integer_value(scalar, &format!("ulong({a}) {symbol} ulong({b})"));
    }
    if scalar.is_signed() && matches!(op, Add | Sub | Mul) {
        return integer_value(scalar, &format!("uint(int({a})) {symbol} uint(int({b}))"));
    }
    if matches!(op, Shl | Shr) {
        let count = format!("(uint({b}) & {}u)", scalar.bits() - 1);
        let value = if scalar == Scalar::Isize && op == Shr {
            format!(
                "(ulong({a}) >> {count}) | (((0ul - ulong({a} < 0)) << ((64u - {count}) & 63u)) & (0ul - ulong({count} != 0u)))"
            )
        } else if scalar.is_signed() && op == Shr {
            format!(
                "(uint(int({a})) >> {count}) | (((0u - uint({a} < 0)) << ((32u - {count}) & 31u)) & (0u - uint({count} != 0u)))"
            )
        } else {
            let width = if scalar.bits() == 64 { "ulong" } else { "uint" };
            format!("{width}({a}) {symbol} {count}")
        };
        return integer_value(scalar, &value);
    }
    if scalar.bits() < 32
        && scalar.is_integer()
        && matches!(op, Add | Sub | Mul | BitAnd | BitOr | BitXor)
    {
        return integer_value(scalar, &format!("uint({a}) {symbol} uint({b})"));
    }
    format!("({a} {symbol} {b})")
}
