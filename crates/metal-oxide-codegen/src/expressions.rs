use metal_oxide_ir::*;

pub(crate) fn type_name(ty: Type) -> String {
    match ty {
        Type::Unit | Type::Never => "void".into(),
        Type::Scalar(s) => scalar_name(s).into(),
        Type::Dim3 => "uint3".into(),
        Type::Aggregate(id) => format!("metal_oxide_aggregate_{id}"),
        Type::Checked(s) => format!("metal_oxide_checked_{}", s.name()),
        Type::Buffer {
            element,
            access,
            address_space,
        } => {
            let space = match address_space {
                AddressSpace::Device => "device",
                AddressSpace::Threadgroup => "threadgroup",
            };
            let qualifier = if access == Access::Read { "const " } else { "" };
            let atomic = if access == Access::Atomic {
                "atomic_"
            } else {
                ""
            };
            format!("{space} {qualifier}{atomic}{} *", type_name(element.ty()))
        }
    }
}

pub(crate) fn scalar_name(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::Bool => "bool",
        Scalar::F32 => "float",
        Scalar::F16 => "half",
        Scalar::U32 => "uint",
        Scalar::I32 => "int",
        Scalar::U8 => "uchar",
        Scalar::U16 => "ushort",
        Scalar::I8 => "char",
        Scalar::I16 => "short",
        Scalar::Usize => "ulong",
    }
}

fn aggregate_value(module: &Module, id: usize, fields: Vec<String>) -> String {
    let fields = fields.join(", ");
    let fields = match module.types.get(id).unwrap() {
        Aggregate::Array { .. } => format!("{{{fields}}}"),
        Aggregate::Record { .. } | Aggregate::Tuple(_) => fields,
    };
    format!("metal_oxide_aggregate_{id}{{{fields}}}")
}

pub(crate) fn aggregate_field(module: &Module, id: usize, value: &str, field: u32) -> String {
    match module.types.get(id).unwrap() {
        Aggregate::Array { .. } => format!("{value}.elements[{field}]"),
        Aggregate::Record { .. } | Aggregate::Tuple(_) => format!("{value}.f{field}"),
    }
}

pub(crate) fn place(module: &Module, function: &Function, value: &Place) -> String {
    let mut ty = function.locals[value.local];
    let mut result = format!("v{}", value.local);
    for projection in &value.projection {
        if let Projection::Index(index) = *projection {
            result = format!("{result}.elements[v{index}]");
            let Type::Aggregate(id) = ty else {
                unreachable!()
            };
            let Aggregate::Array { element, .. } = module.types.get(id).unwrap() else {
                unreachable!()
            };
            ty = *element;
            continue;
        }
        let Projection::Field(field) = *projection else {
            unreachable!()
        };
        result = match ty {
            Type::Aggregate(id) => aggregate_field(module, id, &result, field),
            Type::Dim3 => format!("{result}.{}", ["x", "y", "z"][field as usize]),
            Type::Checked(_) => format!("{result}.{}", ["value", "overflow"][field as usize]),
            _ => unreachable!("validated projection"),
        };
        ty = module.types.field(ty, field).unwrap();
    }
    result
}

pub(crate) fn operand(module: &Module, function: &Function, value: &Operand) -> String {
    match value {
        Operand::Place(value) if function.locals[value.local] == Type::Unit => String::new(),
        Operand::Place(value) => place(module, function, value),
        Operand::AggregateConstant { ty, fields } => {
            let fields = fields
                .iter()
                .map(|value| operand(module, function, value))
                .collect::<Vec<_>>();
            match ty {
                Type::Aggregate(id) => aggregate_value(module, *id, fields),
                _ => format!("{}{{{}}}", type_name(*ty), fields.join(", ")),
            }
        }
        Operand::Constant(value) => match value {
            Constant::Unit => String::new(),
            Constant::Bool(v) => v.to_string(),
            Constant::F32(bits) => format!("as_type<float>(0x{bits:08x}u)"),
            Constant::F16(bits) => format!("as_type<half>(ushort({bits}u))"),
            Constant::U32(v) => format!("{v}u"),
            Constant::I32(v) => format!("as_type<int>(0x{:08x}u)", *v as u32),
            Constant::U8(v) => format!("uchar({v}u)"),
            Constant::U16(v) => format!("ushort({v}u)"),
            Constant::I8(v) => format!("as_type<char>(uchar({}u))", *v as u8),
            Constant::I16(v) => format!("as_type<short>(ushort({}u))", *v as u16),
            Constant::Usize(v) => format!("{v}ul"),
        },
    }
}

pub(crate) fn expression(
    module: &Module,
    function: &Function,
    value: &Expression,
    source: &SourceLocation,
) -> Result<String, Error> {
    let op = |v| operand(module, function, v);
    let ty = |v| operand_type(module, function, v, source);
    Ok(match value {
        Expression::Use(v) => op(v),
        Expression::Bitcast(v, to) => format!("as_type<{}>({})", scalar_name(*to), op(v)),
        Expression::Math {
            op: operation,
            arguments,
        } => crate::floating::math(*operation, arguments.iter().map(op).collect()),
        Expression::Checked {
            scalar,
            value,
            overflow,
        } => format!(
            "{}{{{}, {}}}",
            type_name(Type::Checked(*scalar)),
            op(value),
            op(overflow)
        ),
        Expression::Aggregate { ty, fields } => {
            aggregate_value(module, *ty, fields.iter().map(op).collect())
        }
        Expression::AggregateUpdate {
            aggregate,
            field,
            value,
        } => {
            let Type::Aggregate(id) = ty(aggregate)? else {
                unreachable!("validated aggregate update")
            };
            format!(
                "metal_oxide_update_{id}_{field}({}, {})",
                op(aggregate),
                op(value)
            )
        }
        Expression::SimdCoordinate(builtin) => format!(
            "metal_oxide_ctx.simd_{}",
            match builtin {
                SimdBuiltin::Lane => "lane",
                SimdBuiltin::Size => "size",
                SimdBuiltin::Group => "group",
                SimdBuiltin::Count => "count",
            }
        ),
        Expression::SimdSum(value) => format!("simd_sum({})", op(value)),
        Expression::SimdShuffle { value, lane } => {
            format!("simd_shuffle({}, {})", op(value), op(lane))
        }
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
        Expression::Cast(v, to) => crate::arithmetic::cast(&op(v), ty(v)?, *to, source)?,
        Expression::Unary(operation, v) => crate::arithmetic::unary(*operation, &op(v), ty(v)?),
        Expression::Binary(operation, a, b) => {
            crate::arithmetic::binary(*operation, &op(a), &op(b), ty(a)?, source)?
        }
    })
}
