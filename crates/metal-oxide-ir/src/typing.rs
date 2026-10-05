use crate::*;

pub fn operand_type(
    function: &Function,
    operand: &Operand,
    source: &SourceLocation,
) -> Result<Type, Error> {
    match operand {
        Operand::Constant(value) => Ok(value.ty()),
        Operand::Place { local, field } => {
            let ty = *function
                .locals
                .get(*local)
                .ok_or_else(|| Error::new(source, "invalid local reference"))?;
            match (ty, field) {
                (Type::Never, _) => Err(Error::new(source, "never-typed values cannot be read")),
                (_, None) => Ok(ty),
                (Type::Dim3, Some(0..=2)) => Ok(Type::Scalar(Scalar::U32)),
                (Type::Checked(scalar), Some(0)) => Ok(Type::Scalar(scalar)),
                (Type::Checked(_), Some(1)) => Ok(Type::Scalar(Scalar::Bool)),
                _ => Err(Error::new(source, "unsupported field projection")),
            }
        }
    }
}

pub(crate) fn expression_type(
    module: &Module,
    function: &Function,
    expression: &Expression,
    source: &SourceLocation,
) -> Result<Type, Error> {
    let ty = |v| operand_type(function, v, source);
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
                (UnaryOp::Neg, Type::Scalar(Scalar::F32 | Scalar::I32))
                    | (
                        UnaryOp::Not,
                        Type::Scalar(Scalar::Bool | Scalar::U32 | Scalar::I32)
                    )
            );
            if valid {
                Ok(t)
            } else {
                Err(Error::new(source, "invalid unary operation type"))
            }
        }
        Expression::Cast(v, to) => {
            if matches!(ty(v)?, Type::Scalar(_)) {
                Ok(Type::Scalar(*to))
            } else {
                Err(Error::new(source, "casts require scalar types"))
            }
        }
        Expression::Coordinates(_) => Ok(Type::Dim3),
        Expression::ThreadgroupAlloc {
            element, length, ..
        } => {
            if *length == 0 || *element == Scalar::Bool || length.checked_mul(4).is_none() {
                return Err(Error::new(source, "invalid threadgroup allocation"));
            }
            Ok(Type::Buffer {
                element: *element,
                access: Access::ReadWrite,
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
                } => Ok(Type::Scalar(element)),
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
                    require(ty(value)?, Type::Scalar(element))?;
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
        Expression::AtomicAdd {
            buffer,
            index,
            value,
        } => {
            require(ty(index)?, Type::Scalar(Scalar::U32))?;
            match ty(buffer)? {
                Type::Buffer {
                    element: element @ (Scalar::U32 | Scalar::I32),
                    access: Access::Atomic,
                    address_space: AddressSpace::Device,
                } => {
                    require(ty(value)?, Type::Scalar(element))?;
                    Ok(Type::Scalar(element))
                }
                _ => Err(Error::new(
                    source,
                    "atomic addition requires an i32/u32 atomic buffer",
                )),
            }
        }
    }
}

fn binary_type(op: BinaryOp, a: Type, b: Type, source: &SourceLocation) -> Result<Type, Error> {
    use BinaryOp::*;
    use Scalar::*;
    let integer = matches!(a, Type::Scalar(U32 | I32));
    let number = integer || a == Type::Scalar(F32);
    let boolean = a == Type::Scalar(Bool);
    let valid = match op {
        Shl | Shr => integer && matches!(b, Type::Scalar(U32 | I32)),
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
