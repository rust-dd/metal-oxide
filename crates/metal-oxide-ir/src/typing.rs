use crate::*;

pub fn place_type(
    module: &Module,
    function: &Function,
    place: &Place,
    source: &SourceLocation,
) -> Result<Type, Error> {
    let mut ty = *function
        .locals
        .get(place.local)
        .ok_or_else(|| Error::new(source, "invalid local reference"))?;
    if ty == Type::Never {
        return Err(Error::new(source, "never-typed values cannot be read"));
    }
    for projection in &place.projection {
        ty = match *projection {
            Projection::Field(field) => module.types.field(ty, field),
            Projection::Index(index) => {
                if function.locals.get(index) != Some(&Type::Scalar(Scalar::Usize)) {
                    return Err(Error::new(source, "array indices require a usize local"));
                }
                match ty {
                    Type::Aggregate(id) => match module.types.get(id) {
                        Some(Aggregate::Array { element, .. }) => Some(*element),
                        _ => None,
                    },
                    _ => None,
                }
            }
        }
        .ok_or_else(|| Error::new(source, "invalid aggregate field projection"))?;
    }
    Ok(ty)
}

pub fn operand_type(
    module: &Module,
    function: &Function,
    operand: &Operand,
    source: &SourceLocation,
) -> Result<Type, Error> {
    match operand {
        Operand::Constant(value) => Ok(value.ty()),
        Operand::AggregateConstant { ty, fields } => {
            if fields.len() != module.types.field_count(*ty) || fields.is_empty() {
                return Err(Error::new(source, "invalid aggregate constant shape"));
            }
            for (index, value) in fields.iter().enumerate() {
                if matches!(value, Operand::Place(_))
                    || operand_type(module, function, value, source)?
                        != module.types.field(*ty, index as u32).unwrap()
                {
                    return Err(Error::new(
                        source,
                        "aggregate constants require matching constant components",
                    ));
                }
            }
            Ok(*ty)
        }
        Operand::Place(place) => place_type(module, function, place, source),
    }
}

pub(crate) fn expression_type(
    module: &Module,
    function: &Function,
    expression: &Expression,
    source: &SourceLocation,
) -> Result<Type, Error> {
    let ty = |v| operand_type(module, function, v, source);
    let require = |actual, expected| {
        if actual == expected {
            Ok(())
        } else {
            Err(Error::new(
                source,
                format!("type mismatch: expected {expected:?}, found {actual:?}"),
            ))
        }
    };
    match expression {
        Expression::Use(v) => ty(v),
        Expression::Binary(op, a, b) => binary_type(*op, ty(a)?, ty(b)?, source),
        Expression::Unary(op, v) => {
            let t = ty(v)?;
            let valid = matches!(
                (op, t),
                (
                    UnaryOp::Neg,
                    Type::Scalar(Scalar::F32 | Scalar::I8 | Scalar::I16 | Scalar::I32)
                ) | (
                    UnaryOp::Not,
                    Type::Scalar(
                        Scalar::Bool
                            | Scalar::U32
                            | Scalar::I32
                            | Scalar::U8
                            | Scalar::U16
                            | Scalar::I8
                            | Scalar::I16
                            | Scalar::Usize
                    )
                )
            );
            if valid {
                Ok(t)
            } else {
                Err(Error::new(source, "invalid unary operation type"))
            }
        }
        Expression::Cast(v, to) => {
            let from = ty(v)?;
            if matches!(from, Type::Scalar(Scalar::F16)) || *to == Scalar::F16 {
                if matches!(
                    (from, to),
                    (Type::Scalar(Scalar::F16), Scalar::F32)
                        | (Type::Scalar(Scalar::F32), Scalar::F16)
                ) {
                    return Ok(Type::Scalar(*to));
                }
                return Err(Error::new(
                    source,
                    "f16 storage requires an explicit f32 conversion",
                ));
            }
            if matches!(from, Type::Scalar(_)) {
                Ok(Type::Scalar(*to))
            } else {
                Err(Error::new(source, "casts require scalar types"))
            }
        }
        Expression::Coordinates(_) => Ok(Type::Dim3),
        Expression::Bitcast(value, to) => {
            if matches!(
                (ty(value)?, to),
                (Type::Scalar(Scalar::F32), Scalar::U32)
                    | (Type::Scalar(Scalar::U32), Scalar::F32)
                    | (Type::Scalar(Scalar::F16), Scalar::U16)
                    | (Type::Scalar(Scalar::U16), Scalar::F16)
            ) {
                Ok(Type::Scalar(*to))
            } else {
                Err(Error::new(source, "unsupported scalar bitcast"))
            }
        }
        Expression::Math { op, arguments } => {
            if arguments.len() != op.arity() {
                return Err(Error::new(source, "invalid math argument count"));
            }
            for argument in arguments {
                require(ty(argument)?, Type::Scalar(Scalar::F32))?;
            }
            Ok(Type::Scalar(Scalar::F32))
        }
        Expression::Checked {
            scalar,
            value,
            overflow,
        } => {
            if !scalar.is_integer() {
                return Err(Error::new(source, "checked tuple requires integer value"));
            }
            require(ty(value)?, Type::Scalar(*scalar))?;
            require(ty(overflow)?, Type::Scalar(Scalar::Bool))?;
            Ok(Type::Checked(*scalar))
        }
        Expression::Aggregate { ty: id, fields } => {
            let types = module
                .types
                .get(*id)
                .ok_or_else(|| Error::new(source, "invalid aggregate type"))?;
            if fields.len() != types.len() {
                return Err(Error::new(source, "aggregate field count mismatch"));
            }
            for (index, value) in fields.iter().enumerate() {
                require(ty(value)?, types.field(index as u32).unwrap())?;
            }
            Ok(Type::Aggregate(*id))
        }
        Expression::SimdCoordinate(_) => Ok(Type::Scalar(Scalar::U32)),
        Expression::SimdSum(value) => {
            require(ty(value)?, Type::Scalar(Scalar::F32))?;
            Ok(Type::Scalar(Scalar::F32))
        }
        Expression::SimdShuffle { value, lane } => {
            require(ty(value)?, Type::Scalar(Scalar::F32))?;
            require(ty(lane)?, Type::Scalar(Scalar::U32))?;
            Ok(Type::Scalar(Scalar::F32))
        }
        Expression::ThreadgroupAlloc {
            element,
            length,
            access,
            ..
        } => {
            if !matches!(access, Access::ReadWrite | Access::Atomic)
                || (*access == Access::Atomic && !matches!(element, Scalar::U32 | Scalar::I32))
                || *length == 0
                || matches!(element, Scalar::Bool | Scalar::Usize)
                || length.checked_mul(element.bits() / 8).is_none()
            {
                return Err(Error::new(source, "invalid threadgroup allocation"));
            }
            Ok(Type::Buffer {
                element: Element::Scalar(*element),
                access: *access,
                address_space: AddressSpace::Threadgroup,
            })
        }
        Expression::ThreadgroupBarrier => Ok(Type::Unit),
        Expression::Dim3(values) => {
            for v in values {
                require(ty(v)?, Type::Scalar(Scalar::U32))?;
            }
            Ok(Type::Dim3)
        }
        Expression::BufferLoad { buffer, index } => {
            require(ty(index)?, Type::Scalar(Scalar::U32))?;
            match ty(buffer)? {
                Type::Buffer {
                    element,
                    access: Access::Read | Access::ReadWrite,
                    ..
                } => Ok(element.ty()),
                _ => Err(Error::new(source, "buffer load requires a read buffer")),
            }
        }
        Expression::BufferStore {
            buffer,
            index,
            value,
        } => {
            require(ty(index)?, Type::Scalar(Scalar::U32))?;
            match ty(buffer)? {
                Type::Buffer {
                    element,
                    access: Access::Write | Access::ReadWrite,
                    ..
                } => {
                    require(ty(value)?, element.ty())?;
                    Ok(Type::Unit)
                }
                _ => Err(Error::new(source, "buffer store requires a write buffer")),
            }
        }
        Expression::Call {
            function: id,
            arguments,
        } => {
            let callee = module
                .functions
                .get(*id)
                .ok_or_else(|| Error::new(source, "invalid function reference"))?;
            if callee.kernel {
                return Err(Error::new(
                    source,
                    "calling a kernel entrypoint is unsupported",
                ));
            }
            if arguments.len() != callee.parameters {
                return Err(Error::new(source, "function argument count mismatch"));
            }
            for (argument, expected) in arguments.iter().zip(&callee.locals[1..=callee.parameters])
            {
                require(ty(argument)?, *expected)?;
            }
            Ok(callee.locals[0])
        }
        Expression::Atomic { op, arguments } => {
            crate::atomic::atomic_type(module, function, *op, arguments, source)
        }
    }
}

fn binary_type(op: BinaryOp, a: Type, b: Type, source: &SourceLocation) -> Result<Type, Error> {
    use BinaryOp::*;
    use Scalar::*;
    let integer = matches!(a, Type::Scalar(s) if s.is_integer());
    let number = integer || a == Type::Scalar(F32);
    let boolean = a == Type::Scalar(Bool);
    let valid = match op {
        Shl | Shr => integer && matches!(b, Type::Scalar(s) if s.is_integer()),
        BitAnd | BitOr | BitXor => (integer || boolean) && a == b,
        Eq | Ne | Lt | Le | Gt | Ge => (number || boolean) && a == b,
        AddWithOverflow | SubWithOverflow | MulWithOverflow => integer && a == b,
        Add | Sub | Mul | Div | Rem => number && a == b,
    };
    if !valid {
        return Err(Error::new(source, "invalid binary operation types"));
    }
    Ok(match op {
        Eq | Ne | Lt | Le | Gt | Ge => Type::Scalar(Bool),
        AddWithOverflow | SubWithOverflow | MulWithOverflow => {
            let Type::Scalar(s) = a else { unreachable!() };
            Type::Checked(s)
        }
        _ => a,
    })
}
