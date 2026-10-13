use crate::*;

/// What a place denotes: a value, or the elements behind a dereferenced slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Located {
    Value(Type),
    Elements(Pointer),
}

#[derive(Clone, Copy, Debug)]
pub struct PlaceInfo {
    pub located: Located,
    pub space: AddressSpace,
    /// Writes and mutable references need every pointer on the path to be mutable.
    pub writable: bool,
}

pub fn place_info(
    module: &Module,
    function: &Function,
    place: &Place,
    source: &SourceLocation,
) -> Result<PlaceInfo, Error> {
    let ty = *function
        .locals
        .get(place.local)
        .ok_or_else(|| Error::new(source, "invalid local reference"))?;
    if ty == Type::Never {
        return Err(Error::new(source, "never-typed values cannot be read"));
    }
    let mut info = PlaceInfo {
        located: Located::Value(ty),
        space: AddressSpace::Thread,
        writable: true,
    };
    for projection in &place.projection {
        let invalid = match projection {
            Projection::Field(_) => "invalid aggregate field projection",
            Projection::Index(_) => "invalid array or slice index projection",
            Projection::Deref => "dereference requires a pointer",
        };
        info.located = match (*projection, info.located) {
            (Projection::Field(field), Located::Value(ty)) => {
                module.types.field(ty, field).map(Located::Value)
            }
            (Projection::Index(index), located) => {
                if function.locals.get(index) != Some(&Type::Scalar(Scalar::Usize)) {
                    return Err(Error::new(source, "array indices require a usize local"));
                }
                match located {
                    Located::Value(Type::Aggregate(id)) => match module.types.get(id) {
                        Some(Aggregate::Array { element, .. }) => Some(Located::Value(*element)),
                        _ => None,
                    },
                    Located::Elements(slice) => Some(Located::Value(slice.target)),
                    Located::Value(_) => None,
                }
            }
            (Projection::Deref, Located::Value(ty)) => module.types.pointer(ty).map(|pointer| {
                info.space = pointer.space;
                info.writable &= pointer.mutable;
                if pointer.slice {
                    Located::Elements(pointer)
                } else {
                    Located::Value(pointer.target)
                }
            }),
            _ => None,
        }
        .ok_or_else(|| Error::new(source, invalid))?;
    }
    Ok(info)
}

pub fn place_type(
    module: &Module,
    function: &Function,
    place: &Place,
    source: &SourceLocation,
) -> Result<Type, Error> {
    match place_info(module, function, place, source)?.located {
        Located::Value(ty) => Ok(ty),
        Located::Elements(_) => Err(Error::new(source, "slice elements are not one value")),
    }
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

pub fn expression_type(
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
                            | Scalar::U64
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
        Expression::UncheckedCast(value, to) => {
            require(ty(value)?, Type::Scalar(Scalar::F32))?;
            if !to.is_integer() || matches!(to, Scalar::Bool) {
                return Err(Error::new(
                    source,
                    "unchecked conversion requires an integer target",
                ));
            }
            Ok(Type::Scalar(*to))
        }
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
            if *op == MathOp::SinCos {
                return crate::sin_cos_type(&module.types)
                    .ok_or_else(|| Error::new(source, "sin_cos requires an (f32, f32) result"));
            }
            Ok(Type::Scalar(Scalar::F32))
        }
        Expression::Integer { op, arguments } => {
            if arguments.len() != op.arity() {
                return Err(Error::new(
                    source,
                    "invalid integer operation argument count",
                ));
            }
            let scalar = match ty(&arguments[0])? {
                Type::Scalar(scalar) if scalar.is_integer() => scalar,
                _ => {
                    return Err(Error::new(
                        source,
                        "integer operation requires an integer scalar",
                    ));
                }
            };
            match op {
                IntegerOp::Min | IntegerOp::Max => {
                    require(ty(&arguments[1])?, Type::Scalar(scalar))?
                }
                IntegerOp::RotateLeft | IntegerOp::RotateRight => {
                    require(ty(&arguments[1])?, Type::Scalar(Scalar::U32))?
                }
                _ => {}
            }
            Ok(Type::Scalar(match op {
                IntegerOp::LeadingZeros | IntegerOp::TrailingZeros | IntegerOp::CountOnes => {
                    Scalar::U32
                }
                _ => scalar,
            }))
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
        Expression::ThreadIndex => Ok(Type::Scalar(Scalar::Usize)),
        Expression::Simd { op, arguments } => {
            crate::simd::simd_type(module, function, *op, arguments, source)
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
        Expression::Log { format, arguments } => {
            let pieces =
                crate::log_pieces(format).map_err(|message| Error::new(source, message))?;
            let placeholders = pieces
                .iter()
                .filter(|piece| **piece == crate::LogPiece::Argument)
                .count();
            if placeholders != arguments.len() {
                return Err(Error::new(
                    source,
                    format!(
                        "log format takes {placeholders} arguments, got {}",
                        arguments.len()
                    ),
                ));
            }
            for argument in arguments {
                if !matches!(ty(argument)?, Type::Scalar(scalar) if scalar != Scalar::Bool) {
                    return Err(Error::new(
                        source,
                        "log arguments must be integers, f32 or F16",
                    ));
                }
            }
            Ok(Type::Unit)
        }
        Expression::Uninit(ty) => match ty {
            Type::Aggregate(id)
                if matches!(module.types.get(*id), Some(Aggregate::Array { .. })) =>
            {
                Ok(*ty)
            }
            _ => Err(Error::new(source, "uninitialized storage must be an array")),
        },
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
        Expression::Ref { place, mutable } => {
            let Operand::Place(place) = place else {
                return Err(Error::new(source, "references require a place"));
            };
            let info = place_info(module, function, place, source)?;
            if *mutable && !info.writable {
                return Err(Error::new(
                    source,
                    "a mutable reference requires a mutable path",
                ));
            }
            let (target, slice) = match info.located {
                Located::Value(ty) => (ty, false),
                Located::Elements(elements) => (elements.target, true),
            };
            pointer_type(
                module,
                Pointer {
                    target,
                    space: info.space,
                    mutable: *mutable,
                    slice,
                },
                source,
            )
        }
        Expression::Len(slice) => match ty(slice)? {
            Type::Buffer {
                address_space: AddressSpace::Device,
                ..
            } => Ok(Type::Scalar(Scalar::Usize)),
            ty => match module.types.pointer(ty) {
                Some(pointer) if pointer.slice => Ok(Type::Scalar(Scalar::Usize)),
                _ => Err(Error::new(source, "length requires a slice or buffer")),
            },
        },
        Expression::Slice(data) => {
            let slice = match ty(data)? {
                Type::Buffer {
                    element,
                    access: access @ (Access::Read | Access::Write | Access::ReadWrite),
                    address_space: AddressSpace::Device,
                } => Pointer {
                    target: element.ty(),
                    space: AddressSpace::Device,
                    mutable: access != Access::Read,
                    slice: true,
                },
                array => match module.types.pointer(array) {
                    Some(
                        pointer @ Pointer {
                            target: Type::Aggregate(id),
                            slice: false,
                            ..
                        },
                    ) => match module.types.get(id) {
                        Some(Aggregate::Array { element, .. }) => Pointer {
                            target: *element,
                            slice: true,
                            ..pointer
                        },
                        _ => return Err(Error::new(source, "slices require an array or buffer")),
                    },
                    _ => return Err(Error::new(source, "slices require an array or buffer")),
                },
            };
            pointer_type(module, slice, source)
        }
        Expression::Subslice { slice, start, end } => {
            require(ty(start)?, Type::Scalar(Scalar::Usize))?;
            require(ty(end)?, Type::Scalar(Scalar::Usize))?;
            let slice = ty(slice)?;
            match module.types.pointer(slice) {
                Some(pointer) if pointer.slice => Ok(slice),
                _ => Err(Error::new(source, "subslices require a slice")),
            }
        }
    }
}

fn pointer_type(module: &Module, pointer: Pointer, source: &SourceLocation) -> Result<Type, Error> {
    module
        .types
        .find_pointer(pointer)
        .ok_or_else(|| Error::new(source, format!("pointer type {pointer:?} is not interned")))
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
