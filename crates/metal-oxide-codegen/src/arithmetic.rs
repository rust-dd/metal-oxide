use metal_oxide_ir::*;

use crate::expressions::scalar_name;

fn integer_value(scalar: Scalar, value: &str) -> String {
    let name = scalar_name(scalar);
    if scalar.is_signed() {
        let unsigned = match scalar.bits() {
            8 => "uchar",
            16 => "ushort",
            32 => "uint",
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
        (UnaryOp::Neg, Type::Scalar(scalar)) if scalar.is_signed() => {
            integer_value(scalar, &format!("0u - uint(int({value}))"))
        }
        (UnaryOp::Neg, _) => format!("(-{value})"),
        (UnaryOp::Not, Type::Scalar(Scalar::Bool)) => format!("(!{value})"),
        (UnaryOp::Not, Type::Scalar(Scalar::Usize)) => format!("(~{value})"),
        (UnaryOp::Not, Type::Scalar(scalar)) => integer_value(scalar, &format!("~uint({value})")),
        _ => unreachable!("validated unary type"),
    }
}

pub(crate) fn binary(
    op: BinaryOp,
    a: &str,
    b: &str,
    ty: Type,
    _source: &SourceLocation,
) -> Result<String, Error> {
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
    };
    if matches!(op, AddWithOverflow | SubWithOverflow | MulWithOverflow) {
        let Type::Scalar(scalar) = ty else {
            unreachable!()
        };
        let label = match op {
            AddWithOverflow => "add",
            SubWithOverflow => "sub",
            _ => "mul",
        };
        return Ok(format!(
            "metal_oxide_{label}_checked_{}({a}, {b})",
            scalar.name()
        ));
    }
    if ty == Type::Scalar(Scalar::F32) && op == Rem {
        return Ok(format!("fmod({a}, {b})"));
    }
    let Type::Scalar(scalar) = ty else {
        unreachable!("validated binary type")
    };
    if scalar.is_signed() && matches!(op, Add | Sub | Mul) {
        return Ok(integer_value(
            scalar,
            &format!("uint(int({a})) {symbol} uint(int({b}))"),
        ));
    }
    if matches!(op, Shl | Shr) {
        let count = format!("(uint({b}) & {}u)", scalar.bits() - 1);
        let value = if scalar.is_signed() && op == Shr {
            format!(
                "(uint(int({a})) >> {count}) | (((0u - uint({a} < 0)) << ((32u - {count}) & 31u)) & (0u - uint({count} != 0u)))"
            )
        } else {
            let width = if scalar == Scalar::Usize {
                "ulong"
            } else {
                "uint"
            };
            format!("{width}({a}) {symbol} {count}")
        };
        return Ok(integer_value(scalar, &value));
    }
    if scalar.bits() < 32
        && scalar.is_integer()
        && matches!(op, Add | Sub | Mul | BitAnd | BitOr | BitXor)
    {
        return Ok(integer_value(
            scalar,
            &format!("uint({a}) {symbol} uint({b})"),
        ));
    }
    Ok(format!("({a} {symbol} {b})"))
}
