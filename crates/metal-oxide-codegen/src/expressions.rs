use metal_oxide_ir::*;

pub(crate) fn type_name(ty: Type) -> &'static str {
    match ty {
        Type::Unit | Type::Never => "void",
        Type::Scalar(Scalar::Bool) => "bool",
        Type::Scalar(Scalar::F32) => "float",
        Type::Scalar(Scalar::U32) => "uint",
        Type::Scalar(Scalar::I32) => "int",
        Type::Dim3 => "uint3",
        Type::Checked(Scalar::U32) => "metal_oxide_checked_u32",
        Type::Checked(Scalar::I32) => "metal_oxide_checked_i32",
        Type::Buffer {
            element: Scalar::F32,
            address_space: AddressSpace::Threadgroup,
            ..
        } => "threadgroup float *",
        Type::Buffer {
            element: Scalar::U32,
            address_space: AddressSpace::Threadgroup,
            ..
        } => "threadgroup uint *",
        Type::Buffer {
            element: Scalar::I32,
            address_space: AddressSpace::Threadgroup,
            ..
        } => "threadgroup int *",
        Type::Buffer {
            element: Scalar::U32,
            access: Access::Atomic,
            ..
        } => "device atomic_uint *",
        Type::Buffer {
            element: Scalar::I32,
            access: Access::Atomic,
            ..
        } => "device atomic_int *",
        Type::Buffer {
            element: Scalar::F32,
            access: Access::Read,
            ..
        } => "device const float *",
        Type::Buffer {
            element: Scalar::F32,
            access: Access::Write,
            ..
        } => "device float *",
        Type::Buffer {
            element: Scalar::U32,
            access: Access::Read,
            ..
        } => "device const uint *",
        Type::Buffer {
            element: Scalar::U32,
            access: Access::Write,
            ..
        } => "device uint *",
        Type::Buffer {
            element: Scalar::I32,
            access: Access::Read,
            ..
        } => "device const int *",
        Type::Buffer {
            element: Scalar::I32,
            access: Access::Write,
            ..
        } => "device int *",
        _ => unreachable!("IR validation rejects this type"),
    }
}

pub(crate) fn operand(function: &Function, value: &Operand) -> String {
    match value {
        Operand::Place { local, field: None } if function.locals[*local] == Type::Unit => {
            String::new()
        }
        Operand::Place { local, field: None } => format!("v{local}"),
        Operand::Place {
            local,
            field: Some(field),
        } => {
            let field = match function.locals[*local] {
                Type::Dim3 => ["x", "y", "z"][*field as usize],
                Type::Checked(_) => ["value", "overflow"][*field as usize],
                _ => unreachable!("validated field projection"),
            };
            format!("v{local}.{field}")
        }
        Operand::Constant(value) => match value {
            Constant::Unit => String::new(),
            Constant::Bool(v) => v.to_string(),
            Constant::F32(bits) => format!("as_type<float>(0x{bits:08x}u)"),
            Constant::U32(v) => format!("{v}u"),
            Constant::I32(v) => format!("as_type<int>(0x{:08x}u)", *v as u32),
        },
    }
}

pub(crate) fn expression(
    module: &Module,
    function: &Function,
    value: &Expression,
    source: &SourceLocation,
) -> Result<String, Error> {
    let op = |v| operand(function, v);
    let ty = |v| operand_type(function, v, source);
    Ok(match value {
        Expression::Use(v) => op(v),
        Expression::ThreadgroupAlloc { id, .. } => format!("metal_oxide_shared_{id}"),
        Expression::ThreadgroupBarrier => "threadgroup_barrier(mem_flags::mem_threadgroup)".into(),
        Expression::AtomicAdd {
            buffer,
            index,
            value,
        } => format!(
            "atomic_fetch_add_explicit(&{}[{}], {}, memory_order_relaxed)",
            op(buffer),
            op(index),
            op(value)
        ),
        Expression::Coordinates(builtin) => format!(
            "metal_oxide_ctx.{}",
            match builtin {
                Builtin::ThreadIdx => "thread_idx",
                Builtin::BlockIdx => "block_idx",
                Builtin::BlockDim => "block_dim",
                Builtin::GridDim => "grid_dim",
            }
        ),
        Expression::Dim3(values) => format!(
            "uint3({}, {}, {})",
            op(&values[0]),
            op(&values[1]),
            op(&values[2])
        ),
        Expression::BufferLoad { buffer, index } => format!("{}[{}]", op(buffer), op(index)),
        Expression::BufferStore {
            buffer,
            index,
            value,
        } => format!("{}[{}] = {}", op(buffer), op(index), op(value)),
        Expression::Call {
            function: id,
            arguments,
        } => {
            let mut arguments = arguments.iter().map(op).collect::<Vec<_>>();
            arguments.push("metal_oxide_ctx".into());
            debug_assert!(!module.functions[*id].kernel);
            format!("metal_oxide_fn_{id}({})", arguments.join(", "))
        }
        Expression::Cast(v, to) => {
            let from = ty(v)?;
            if from == Type::Scalar(Scalar::F32) && matches!(to, Scalar::U32 | Scalar::I32) {
                return Err(Error::new(
                    source,
                    "float-to-integer casts require saturation support",
                ));
            }
            if (*to == Scalar::Bool && from != Type::Scalar(Scalar::Bool))
                || (from == Type::Scalar(Scalar::Bool) && *to == Scalar::F32)
            {
                return Err(Error::new(source, "unsupported boolean cast"));
            }
            let name = type_name(Type::Scalar(*to));
            if matches!(
                (from, to),
                (Type::Scalar(Scalar::U32), Scalar::I32) | (Type::Scalar(Scalar::I32), Scalar::U32)
            ) {
                format!("as_type<{name}>({})", op(v))
            } else {
                format!("{name}({})", op(v))
            }
        }
        Expression::Unary(operation, v) => match operation {
            UnaryOp::Neg if ty(v)? == Type::Scalar(Scalar::I32) => {
                format!("as_type<int>(0u - as_type<uint>({}))", op(v))
            }
            UnaryOp::Neg => format!("(-{})", op(v)),
            UnaryOp::Not if ty(v)? == Type::Scalar(Scalar::Bool) => format!("(!{})", op(v)),
            UnaryOp::Not => format!("(~{})", op(v)),
        },
        Expression::Binary(operation, a, b) => binary(*operation, &op(a), &op(b), ty(a)?, source)?,
    })
}

fn binary(
    operation: BinaryOp,
    a: &str,
    b: &str,
    ty: Type,
    source: &SourceLocation,
) -> Result<String, Error> {
    use BinaryOp::*;
    let symbol = match operation {
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
    if matches!(
        operation,
        AddWithOverflow | SubWithOverflow | MulWithOverflow
    ) {
        let operation = match operation {
            AddWithOverflow => "add",
            SubWithOverflow => "sub",
            _ => "mul",
        };
        let scalar = if ty == Type::Scalar(Scalar::I32) {
            "i32"
        } else {
            "u32"
        };
        return Ok(format!(
            "metal_oxide_{operation}_checked_{scalar}({a}, {b})"
        ));
    }
    if matches!(operation, Div | Rem) {
        if ty != Type::Scalar(Scalar::F32) {
            return Err(Error::new(
                source,
                "integer division/remainder are not supported yet",
            ));
        }
        if operation == Rem {
            return Ok(format!("fmod({a}, {b})"));
        }
    }
    if ty == Type::Scalar(Scalar::I32) && matches!(operation, Add | Sub | Mul) {
        return Ok(format!(
            "as_type<int>(as_type<uint>({a}) {symbol} as_type<uint>({b}))"
        ));
    }
    if matches!(operation, Shl | Shr) {
        let count = format!("(uint({b}) & 31u)");
        if ty == Type::Scalar(Scalar::I32) {
            return Ok(if operation == Shl {
                format!("as_type<int>(as_type<uint>({a}) << {count})")
            } else {
                format!(
                    "as_type<int>((as_type<uint>({a}) >> {count}) | (((0u - uint({a} < 0)) << ((32u - {count}) & 31u)) & (0u - uint({count} != 0u))))"
                )
            });
        }
        return Ok(format!("({a} {symbol} {count})"));
    }
    Ok(format!("({a} {symbol} {b})"))
}
