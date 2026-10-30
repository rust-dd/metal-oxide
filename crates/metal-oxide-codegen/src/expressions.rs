use crate::layout::{self, Member};
use metal_oxide_ir::*;

pub(crate) fn type_name(ty: Type) -> String {
    match ty {
        Type::Unit | Type::Never => "void".into(),
        Type::Scalar(s) => scalar_name(s).into(),
        Type::Dim3 => "uint3".into(),
        Type::Aggregate(id) => format!("metal_oxide_aggregate_{id}"),
        Type::Checked(s) => format!("metal_oxide_checked_{}", s.name()),
        Type::Pointer(id) => format!("metal_oxide_pointer_{id}"),
        Type::Matrix(s) => format!("simdgroup_matrix<{}, 8, 8>", scalar_name(s)),
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
    let data = data_pointer(pointer);
    if pointer.slice {
        format!("struct metal_oxide_pointer_{id} {{\n    {data} data;\n    ulong length;\n}};\n\n")
    } else {
        format!("typedef {data} metal_oxide_pointer_{id};\n")
    }
}

/// The MSL type of a pointer to the values a pointer or slice views.
fn data_pointer(pointer: &Pointer) -> String {
    let constness = if pointer.mutable { "" } else { "const " };
    format!(
        "{} {constness}{}*",
        space_name(pointer.space),
        type_name(pointer.target)
    )
}

pub(crate) fn scalar_name(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::Bool => "bool",
        Scalar::F32 => "float",
        Scalar::F16 => "half",
        Scalar::BF16 => "bfloat",
        Scalar::U32 => "uint",
        Scalar::I32 => "int",
        Scalar::U8 => "uchar",
        Scalar::U16 => "ushort",
        Scalar::I8 => "char",
        Scalar::I16 => "short",
        Scalar::U64 | Scalar::Usize => "ulong",
        Scalar::I64 | Scalar::Isize => "long",
    }
}

/// A struct value; padding members start zeroed.
fn aggregate_value(module: &Module, id: usize, fields: Vec<String>) -> String {
    match module.types.get(id).unwrap() {
        Aggregate::Array { .. } => {
            format!("metal_oxide_aggregate_{id}{{{{{}}}}}", fields.join(", "))
        }
        // An enum constant lists the tag and every slot.
        Aggregate::Record { .. } | Aggregate::Tuple(_) | Aggregate::Enum { .. } => {
            members(module, id, &fields)
        }
    }
}

/// An enum value: the variant's tag, its fields in their slots, other slots value-initialized.
fn variant_value(module: &Module, id: usize, variant: u32, fields: Vec<String>) -> String {
    let aggregate = module.types.get(id).unwrap();
    let mut values = vec!["{}".to_string(); aggregate.len()];
    values[0] = variant.to_string();
    for (index, value) in fields.into_iter().enumerate() {
        values[aggregate.payload(variant, index as u32).unwrap() as usize] = value;
    }
    members(module, id, &values)
}

/// A struct value from its field values; zero-sized fields have no member.
fn members(module: &Module, id: usize, fields: &[String]) -> String {
    let values = layout::shape(module, id)
        .members
        .iter()
        .map(|member| match *member {
            Member::Field(field) => fields[field as usize].as_str(),
            Member::Padding(_) => "{}",
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("metal_oxide_aggregate_{id}{{{values}}}")
}

pub(crate) fn aggregate_field(module: &Module, id: usize, value: &str, field: u32) -> String {
    match module.types.get(id).unwrap() {
        Aggregate::Array { .. } => format!("{value}.elements[{field}]"),
        Aggregate::Record { .. } | Aggregate::Tuple(_) | Aggregate::Enum { .. } => {
            format!("{value}.f{field}")
        }
    }
}

/// Whether a place holds a zero-sized value, which has no storage to read or write.
pub(crate) fn is_unit(module: &Module, function: &Function, place: &Place) -> bool {
    place_type(module, function, place, &function.source).is_ok_and(|ty| ty == Type::Unit)
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
        Operand::Place(value) if is_unit(module, function, value) => String::new(),
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
            Constant::BF16(bits) => format!("as_type<bfloat>(ushort({bits}u))"),
            Constant::U32(v) => format!("{v}u"),
            Constant::I32(v) => format!("as_type<int>(0x{:08x}u)", *v as u32),
            Constant::U8(v) => format!("uchar({v}u)"),
            Constant::U16(v) => format!("ushort({v}u)"),
            Constant::I8(v) => format!("as_type<char>(uchar({}u))", *v as u8),
            Constant::I16(v) => format!("as_type<short>(ushort({}u))", *v as u16),
            Constant::U64(v) | Constant::Usize(v) => format!("{v}ul"),
            Constant::I64(v) | Constant::Isize(v) => {
                format!("as_type<long>(0x{:016x}ul)", *v as u64)
            }
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
        Expression::Select { condition, yes, no } => {
            format!("({} ? {} : {})", op(condition), op(yes), op(no))
        }
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
        Expression::Variant {
            ty,
            variant,
            fields,
        } => variant_value(module, *ty, *variant, fields.iter().map(op).collect()),
        Expression::SimdCoordinate(builtin) => format!(
            "metal_oxide_ctx.simd_{}",
            match builtin {
                SimdBuiltin::Lane => "lane",
                SimdBuiltin::Size => "size",
                SimdBuiltin::Group => "group",
                SimdBuiltin::Count => "count",
            }
        ),
        Expression::ThreadIndex => "ulong(metal_oxide_ctx.thread_index)".into(),
        Expression::Simd {
            op: operation,
            arguments,
        } => crate::simd::expression(*operation, arguments.iter().map(op).collect()),
        Expression::Matrix {
            op: operation,
            arguments,
        } => {
            let types = arguments.iter().map(ty).collect::<Result<Vec<_>, _>>()?;
            let arguments = arguments.iter().map(op).collect::<Vec<_>>();
            crate::matrix::expression(*operation, &types, &arguments)
        }
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
            order,
            failure,
            arguments,
        } => crate::atomics::expression(
            *operation,
            *order,
            *failure,
            arguments.iter().map(op).collect(),
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
                    let length = format!("{}ul", module.types.get(id).unwrap().len());
                    format!(
                        "metal_oxide_pointer_{pointer}{{(*{}).elements, {}}}",
                        op(data),
                        slice_length(module, pointer, length)
                    )
                }
            }
        }
        Expression::SliceFromParts { data, length } => {
            let pointer = result_pointer(module, function, value, source)?;
            format!(
                "metal_oxide_pointer_{pointer}{{{}, {}}}",
                op(data),
                slice_length(module, pointer, op(length))
            )
        }
        Expression::PointerCast(pointer, to) => {
            let from = module.types.pointer(ty(pointer)?).unwrap();
            let Type::Pointer(id) = *to else {
                unreachable!("validated pointer cast")
            };
            let target = module.types.pointer(*to).unwrap();
            let value = op(pointer);
            match (from.slice, target.slice) {
                (false, _) => format!("(metal_oxide_pointer_{id})({value})"),
                (true, false) => format!("(metal_oxide_pointer_{id})({value}.data)"),
                (true, true) => format!(
                    "metal_oxide_pointer_{id}{{({})({value}.data), {value}.length}}",
                    data_pointer(&target)
                ),
            }
        }
        Expression::Subslice { slice, start, end } => {
            let pointer = result_pointer(module, function, value, source)?;
            let (slice, start) = (op(slice), op(start));
            let length = format!("{} - {start}", op(end));
            format!(
                "metal_oxide_pointer_{pointer}{{{slice}.data + {start}, {}}}",
                slice_length(module, pointer, length)
            )
        }
        Expression::Call {
            function: id,
            arguments,
        } => {
            let mut emitted = Vec::with_capacity(arguments.len() + 1);
            for argument in arguments {
                if ty(argument)? != Type::Unit {
                    emitted.push(op(argument));
                }
            }
            emitted.push("metal_oxide_ctx".into());
            debug_assert!(!module.functions[*id].kernel);
            format!("metal_oxide_fn_{id}({})", emitted.join(", "))
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
            let operands = ty(a)?;
            let numerator = crate::arithmetic::division_numerator(*operation, op(a), b, operands);
            crate::arithmetic::binary(*operation, &numerator, &op(b), operands)
        }
    })
}

/// The length of a new slice. Apple's GPU compilers read a private array as it was before a
/// loop over it with a known end, so private slices hide their length.
fn slice_length(module: &Module, pointer: usize, length: String) -> String {
    match module
        .types
        .pointer(Type::Pointer(pointer))
        .map(|p| p.space)
    {
        Some(AddressSpace::Thread) => {
            format!("(({length}) ^ ulong({}))", crate::arithmetic::OPAQUE_ZERO)
        }
        _ => length,
    }
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
