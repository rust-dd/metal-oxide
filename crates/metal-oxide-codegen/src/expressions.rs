use metal_oxide_ir::*;

pub(crate) fn type_name(ty: Type) -> String {
    match ty {
        Type::Unit | Type::Never => "void".into(),
        Type::Scalar(s) => scalar_name(s).into(),
        Type::Dim3 => "uint3".into(),
        Type::Aggregate(id) => format!("metal_oxide_aggregate_{id}"),
        Type::Checked(s) => format!("metal_oxide_checked_{}", s.name()),
        Type::Pointer(id) => format!("metal_oxide_pointer_{id}"),
        Type::Buffer {
            element,
            access,
            address_space,
        } => {
            let space = space_name(address_space);
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

pub(crate) fn space_name(space: AddressSpace) -> &'static str {
    match space {
        AddressSpace::Device => "device",
        AddressSpace::Threadgroup => "threadgroup",
        AddressSpace::Thread => "thread",
    }
}

/// A thin pointer is a typedef; a slice is a struct of its data pointer and length.
pub(crate) fn pointer_declaration(id: usize, pointer: &Pointer) -> String {
    let constness = if pointer.mutable { "" } else { "const " };
    let data = format!(
        "{} {constness}{}*",
        space_name(pointer.space),
        type_name(pointer.target)
    );
    if pointer.slice {
        format!("struct metal_oxide_pointer_{id} {{\n    {data} data;\n    ulong length;\n}};\n\n")
    } else {
        format!("typedef {data} metal_oxide_pointer_{id};\n")
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
        Scalar::U64 | Scalar::Usize => "ulong",
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
    let mut located = Located::Value(function.locals[value.local]);
    let mut result = format!("v{}", value.local);
    for projection in &value.projection {
        (result, located) = match (*projection, located) {
            (Projection::Field(field), Located::Value(ty)) => (
                match ty {
                    Type::Aggregate(id) => aggregate_field(module, id, &result, field),
                    Type::Dim3 => format!("{result}.{}", ["x", "y", "z"][field as usize]),
                    Type::Checked(_) => {
                        format!("{result}.{}", ["value", "overflow"][field as usize])
                    }
                    _ => unreachable!("validated projection"),
                },
                Located::Value(module.types.field(ty, field).unwrap()),
            ),
            (Projection::Index(index), Located::Value(Type::Aggregate(id))) => {
                let Some(Aggregate::Array { element, .. }) = module.types.get(id) else {
                    unreachable!("validated projection")
                };
                (
                    format!("{result}.elements[v{index}]"),
                    Located::Value(*element),
                )
            }
            (Projection::Index(index), Located::Elements(slice)) => (
                format!("{result}.data[v{index}]"),
                Located::Value(slice.target),
            ),
            (Projection::Deref, Located::Value(ty)) => {
                let pointer = module.types.pointer(ty).unwrap();
                if pointer.slice {
                    (result, Located::Elements(pointer))
                } else {
                    (format!("(*{result})"), Located::Value(pointer.target))
                }
            }
            _ => unreachable!("validated projection"),
        };
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
            Constant::U64(v) | Constant::Usize(v) => format!("{v}ul"),
        },
    }
}

pub(crate) fn expression(
    module: &Module,
    function: &Function,
    value: &Expression,
    source: &SourceLocation,
    math_mode: metal_oxide_artifact::MathMode,
) -> Result<String, Error> {
    let op = |v| operand(module, function, v);
    let ty = |v| operand_type(module, function, v, source);
    Ok(match value {
        Expression::Use(v) => op(v),
        Expression::Bitcast(v, to) => format!("as_type<{}>({})", scalar_name(*to), op(v)),
        Expression::UncheckedCast(v, to) => format!("{}({})", scalar_name(*to), op(v)),
        Expression::Math {
            op: operation,
            arguments,
        } => crate::floating::math(*operation, arguments.iter().map(op).collect(), math_mode),
        Expression::Integer {
            op: operation,
            arguments,
        } => match ty(&arguments[0])? {
            Type::Scalar(scalar) => {
                crate::integer::integer(*operation, scalar, arguments.iter().map(op).collect())
            }
            _ => {
                return Err(Error::new(
                    source,
                    "integer operation requires a scalar operand",
                ));
            }
        },
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
        Expression::SimdCoordinate(builtin) => format!(
            "metal_oxide_ctx.simd_{}",
            match builtin {
                SimdBuiltin::Lane => "lane",
                SimdBuiltin::Size => "size",
                SimdBuiltin::Group => "group",
                SimdBuiltin::Count => "count",
            }
        ),
        Expression::Simd {
            op: operation,
            arguments,
        } => crate::simd::expression(*operation, arguments.iter().map(op).collect()),
        Expression::ThreadgroupAlloc { id, .. } => format!("metal_oxide_shared_{id}"),
        Expression::ThreadgroupBarrier => "threadgroup_barrier(mem_flags::mem_threadgroup)".into(),
        Expression::Uninit(_) => {
            return Err(Error::new(
                source,
                "uninitialized storage has no value expression",
            ));
        }
        Expression::Atomic {
            op: operation,
            arguments,
        } => crate::atomics::expression(*operation, arguments.iter().map(op).collect()),
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
        Expression::Ref { place: target, .. } => {
            let Operand::Place(target) = target else {
                unreachable!("validated reference")
            };
            reference(module, function, value, target, source)?
        }
        Expression::Len(data) => match ty(data)? {
            Type::Buffer { .. } => buffer_length(data),
            _ => format!("{}.length", op(data)),
        },
        Expression::Slice(data) => {
            let pointer = result_pointer(module, function, value, source)?;
            match ty(data)? {
                Type::Buffer { .. } => format!(
                    "metal_oxide_pointer_{pointer}{{{}, {}}}",
                    op(data),
                    buffer_length(data)
                ),
                array => {
                    let Some(Type::Aggregate(id)) = module.types.pointer(array).map(|p| p.target)
                    else {
                        unreachable!("validated slice")
                    };
                    format!(
                        "metal_oxide_pointer_{pointer}{{(*{}).elements, {}ul}}",
                        op(data),
                        module.types.get(id).unwrap().len()
                    )
                }
            }
        }
        Expression::Subslice { slice, start, end } => {
            let pointer = result_pointer(module, function, value, source)?;
            let (slice, start) = (op(slice), op(start));
            format!(
                "metal_oxide_pointer_{pointer}{{{slice}.data + {start}, {} - {start}}}",
                op(end)
            )
        }
        Expression::Call {
            function: id,
            arguments,
        } => {
            let mut arguments = arguments.iter().map(op).collect::<Vec<_>>();
            arguments.push("metal_oxide_ctx".into());
            debug_assert!(!module.functions[*id].kernel);
            format!("metal_oxide_fn_{id}({})", arguments.join(", "))
        }
        Expression::Log { format, arguments } => {
            let arguments = arguments
                .iter()
                .map(|argument| match ty(argument)? {
                    Type::Scalar(scalar) => Ok((op(argument), scalar)),
                    _ => Err(Error::new(source, "log arguments must be scalars")),
                })
                .collect::<Result<Vec<_>, Error>>()?;
            crate::log::expression(format, &arguments, source)?
        }
        Expression::Cast(v, to) => crate::arithmetic::cast(&op(v), ty(v)?, *to, source)?,
        Expression::Unary(operation, v) => crate::arithmetic::unary(*operation, &op(v), ty(v)?),
        Expression::Binary(operation, a, b) => {
            crate::arithmetic::binary(*operation, &op(a), &op(b), ty(a)?)
        }
    })
}

fn result_pointer(
    module: &Module,
    function: &Function,
    value: &Expression,
    source: &SourceLocation,
) -> Result<usize, Error> {
    match expression_type(module, function, value, source)? {
        Type::Pointer(id) => Ok(id),
        _ => unreachable!("validated pointer expression"),
    }
}

/// `&place`; a reborrowed slice is rebuilt when its mutability, and so its MSL type, changes.
fn reference(
    module: &Module,
    function: &Function,
    value: &Expression,
    target: &Place,
    source: &SourceLocation,
) -> Result<String, Error> {
    let emitted = place(module, function, target);
    if !matches!(
        place_info(module, function, target, source)?.located,
        Located::Elements(_)
    ) {
        return Ok(format!("&{emitted}"));
    }
    let pointer = result_pointer(module, function, value, source)?;
    let slice = Place {
        local: target.local,
        projection: target.projection[..target.projection.len() - 1].to_vec(),
    };
    Ok(
        if place_type(module, function, &slice, source)? == Type::Pointer(pointer) {
            emitted
        } else {
            format!("metal_oxide_pointer_{pointer}{{{emitted}.data, {emitted}.length}}")
        },
    )
}

/// The bound length of a kernel buffer parameter, which validation requires the operand to be.
fn buffer_length(buffer: &Operand) -> String {
    let Operand::Place(place) = buffer else {
        unreachable!("validated buffer length")
    };
    format!("metal_oxide_lengths[{}]", place.local - 1)
}
