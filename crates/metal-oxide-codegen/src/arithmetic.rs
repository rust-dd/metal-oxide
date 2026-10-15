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
    Ok(if to.is_integer() {
        integer_value(to, value)
    } else {
        format!("{}({value})", scalar_name(to))
    })
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

/// Metal divides a constant `uint` numerator from 0xFFFFFF80 to 0xFFFFFFFD by a variable through a
/// float estimate that overflows; subtracting the divisor first keeps it off that path.
pub(crate) const GUARDED_DIVISION: &str =
    "static inline uint metal_oxide_udiv_u32(uint a, uint b) {
    uint borrow = a >= 0xffffff80u && a >= b ? b : 0u;
    return (a - borrow) / b + uint(borrow != 0u);
}

static inline uint metal_oxide_urem_u32(uint a, uint b) {
    uint borrow = a >= 0xffffff80u && a >= b ? b : 0u;
    return (a - borrow) % b;
}

";

/// Whether `a / b` or `a % b` goes through [`GUARDED_DIVISION`]: a `u32` division by a variable,
/// unless the numerator is a smaller constant. Metal folds values that our IR cannot see.
pub(crate) fn guarded_division(op: BinaryOp, a: &Operand, b: &Operand, ty: Type) -> bool {
    matches!(op, BinaryOp::Div | BinaryOp::Rem)
        && ty == Type::Scalar(Scalar::U32)
        && !matches!(b, Operand::Constant(_))
        && !matches!(a, Operand::Constant(Constant::U32(value)) if *value < 0xffff_ff80)
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
