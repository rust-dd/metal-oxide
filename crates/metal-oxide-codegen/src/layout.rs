use metal_oxide_ir::{Aggregate, Module, Type};

/// A member of a record or tuple struct, in declaration order.
pub(crate) enum Member {
    Field(u32),
    /// Bytes Metal would otherwise pad implicitly in a value buffers can hold. Zeroed members make
    /// a stored record cover every byte, so consecutive records fill whole cache lines.
    Padding(u64),
}

/// The members, size and alignment Metal gives a struct.
pub(crate) struct Shape {
    pub(crate) members: Vec<Member>,
    pub(crate) size: u64,
    pub(crate) alignment: u64,
}

pub(crate) fn shape(module: &Module, id: usize) -> Shape {
    let aggregate = module.types.get(id).unwrap();
    // Only values buffers can hold are stored; explicit padding elsewhere hinders scalarization.
    let stored = module.types.is_abi_value(Type::Aggregate(id));
    if let Aggregate::Array { element, length } = aggregate {
        let (size, alignment) = size_alignment(module, *element);
        return Shape {
            members: Vec::new(),
            size: size * u64::from(*length),
            alignment,
        };
    }
    let mut members = Vec::new();
    let (mut offset, mut alignment) = (0_u64, 1);
    for (field, ty) in aggregate.component_types().enumerate() {
        let (size, field_alignment) = size_alignment(module, ty);
        let start = offset.next_multiple_of(field_alignment);
        if stored && start > offset {
            members.push(Member::Padding(start - offset));
        }
        members.push(Member::Field(field as u32));
        offset = start + size;
        alignment = alignment.max(field_alignment);
    }
    // An empty C++ struct still occupies one byte.
    let size = offset.max(1).next_multiple_of(alignment);
    if stored && size > offset {
        members.push(Member::Padding(size - offset));
    }
    Shape {
        members,
        size,
        alignment,
    }
}

fn size_alignment(module: &Module, ty: Type) -> (u64, u64) {
    match ty {
        Type::Scalar(scalar) => {
            let size = u64::from(scalar.bits().div_ceil(8));
            (size, size)
        }
        Type::Dim3 => (16, 16),
        Type::Checked(scalar) => {
            let (size, alignment) = size_alignment(module, Type::Scalar(scalar));
            ((size + 1).next_multiple_of(alignment), alignment)
        }
        Type::Pointer(_)
            if module
                .types
                .pointer(ty)
                .is_some_and(|pointer| pointer.slice) =>
        {
            (16, 8)
        }
        Type::Pointer(_) => (8, 8),
        Type::Aggregate(id) => {
            let shape = shape(module, id);
            (shape.size, shape.alignment)
        }
        Type::Unit | Type::Never | Type::Buffer { .. } => unreachable!("not a struct member"),
    }
}
