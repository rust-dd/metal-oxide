use crate::{Access, Error, SourceLocation};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    Bool,
    F32,
    F16,
    BF16,
    U32,
    U64,
    I32,
    I64,
    U8,
    U16,
    I8,
    I16,
    /// Device pointer-width integer, restricted to internal values.
    Usize,
    /// Signed device pointer-width integer, restricted to internal values.
    Isize,
}

impl Scalar {
    pub const fn abi(self) -> Option<metal_oxide_artifact::Scalar> {
        use metal_oxide_artifact::Scalar as Abi;
        Some(match self {
            Self::F32 => Abi::F32,
            Self::F16 => Abi::F16,
            Self::BF16 => Abi::BF16,
            Self::U32 => Abi::U32,
            Self::U64 => Abi::U64,
            Self::I32 => Abi::I32,
            Self::I64 => Abi::I64,
            Self::U8 => Abi::U8,
            Self::U16 => Abi::U16,
            Self::I8 => Abi::I8,
            Self::I16 => Abi::I16,
            Self::Bool | Self::Usize | Self::Isize => return None,
        })
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32)
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::U8 | Self::I8 => 8,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 16,
            Self::Bool => 1,
            Self::F32 | Self::U32 | Self::I32 => 32,
            Self::U64 | Self::I64 | Self::Usize | Self::Isize => 64,
        }
    }

    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::U8
                | Self::U16
                | Self::U32
                | Self::U64
                | Self::I32
                | Self::I64
                | Self::I8
                | Self::I16
                | Self::Usize
                | Self::Isize
        )
    }

    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::Isize
        )
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::Usize => "usize",
            Self::Isize => "isize",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AddressSpace {
    Device,
    Threadgroup,
    /// Locals and temporaries, private to one thread.
    Thread,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Element {
    Scalar(Scalar),
    Aggregate(usize),
}

impl Element {
    pub const fn ty(self) -> Type {
        match self {
            Self::Scalar(s) => Type::Scalar(s),
            Self::Aggregate(id) => Type::Aggregate(id),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Unit,
    Never,
    Scalar(Scalar),
    Dim3,
    Aggregate(usize),
    Buffer {
        element: Element,
        access: Access,
        address_space: AddressSpace,
    },
    /// Integer value and its overflow flag, in that order.
    Checked(Scalar),
    /// A reference or slice interned in the type table.
    Pointer(usize),
    /// An 8x8 SIMD-group matrix of f32, F16 or BF16 elements.
    Matrix(Scalar),
}

/// A reference into one address space; a slice also carries its length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pointer {
    /// The referenced value, or the element type of a slice.
    pub target: Type,
    pub space: AddressSpace,
    pub mutable: bool,
    pub slice: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Aggregate {
    Record {
        name: String,
        fields: Vec<RecordField>,
    },
    Tuple(Vec<Type>),
    Array {
        element: Type,
        length: u32,
    },
    /// A tag numbering the variants, then the payload slots: each variant's fields take the
    /// first slots of their type it does not use yet, so variants share storage. Its fields
    /// are the tag, then the slots.
    Enum {
        name: String,
        /// Each variant's field types; a variant's discriminant is its index.
        variants: Vec<Vec<Type>>,
        slots: Vec<Type>,
        /// The slot of each variant's fields.
        payload: Vec<Vec<u32>>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordField {
    pub name: String,
    pub ty: Type,
}

impl Aggregate {
    pub fn record(name: impl Into<String>, values: Vec<Type>) -> Self {
        Self::Record {
            name: name.into(),
            fields: values
                .into_iter()
                .enumerate()
                .map(|(index, ty)| RecordField {
                    name: format!("f{index}"),
                    ty,
                })
                .collect(),
        }
    }

    /// An enum whose variants hold `variants`' field types.
    pub fn enumeration(name: impl Into<String>, variants: Vec<Vec<Type>>) -> Self {
        let mut slots = Vec::<Type>::new();
        let payload = variants
            .iter()
            .map(|fields| {
                let mut used = Vec::<u32>::new();
                fields
                    .iter()
                    .map(|&ty| {
                        let slot = (0..slots.len() as u32)
                            .find(|slot| slots[*slot as usize] == ty && !used.contains(slot))
                            .unwrap_or_else(|| {
                                slots.push(ty);
                                slots.len() as u32 - 1
                            });
                        used.push(slot);
                        slot
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        Self::Enum {
            name: name.into(),
            variants,
            slots,
            payload,
        }
    }

    /// The narrowest unsigned integer that numbers an enum's variants.
    pub fn tag(&self) -> Option<Scalar> {
        let Self::Enum { variants, .. } = self else {
            return None;
        };
        Some(match variants.len() {
            0..=256 => Scalar::U8,
            257..=65536 => Scalar::U16,
            _ => Scalar::U32,
        })
    }

    /// The field that holds field `field` of an enum's `variant`.
    pub fn payload(&self, variant: u32, field: u32) -> Option<u32> {
        let Self::Enum { payload, .. } = self else {
            return None;
        };
        let slot = payload.get(variant as usize)?.get(field as usize)?;
        Some(slot + 1)
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Record { fields, .. } => fields.len(),
            Self::Tuple(fields) => fields.len(),
            Self::Array { length, .. } => *length as usize,
            Self::Enum { slots, .. } => slots.len() + 1,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn field(&self, index: u32) -> Option<Type> {
        match self {
            Self::Record { fields, .. } => fields.get(index as usize).map(|field| field.ty),
            Self::Tuple(fields) => fields.get(index as usize).copied(),
            Self::Array { element, length } => (index < *length).then_some(*element),
            Self::Enum { slots, .. } => match index {
                0 => self.tag().map(Type::Scalar),
                _ => slots.get(index as usize - 1).copied(),
            },
        }
    }

    /// Direct component types; an array contributes its element type once.
    pub fn component_types(&self) -> impl Iterator<Item = Type> + '_ {
        let count = if matches!(self, Self::Array { .. }) {
            1
        } else {
            self.len()
        };
        (0..count).map(move |index| match self {
            Self::Array { element, .. } => *element,
            _ => self.field(index as u32).unwrap(),
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TypeTable {
    aggregates: Vec<Aggregate>,
    pointers: Vec<Pointer>,
}

impl TypeTable {
    /// Interns an owned shape. Nested aggregate IDs must precede their parent.
    pub fn intern(&mut self, aggregate: Aggregate) -> Type {
        let id = match self.aggregates.iter().position(|value| *value == aggregate) {
            Some(id) => id,
            None => {
                let id = self.aggregates.len();
                self.aggregates.push(aggregate);
                id
            }
        };
        Type::Aggregate(id)
    }

    /// Interns a pointer whose target is already interned.
    pub fn intern_pointer(&mut self, pointer: Pointer) -> Type {
        self.find_pointer(pointer).unwrap_or_else(|| {
            self.pointers.push(pointer);
            Type::Pointer(self.pointers.len() - 1)
        })
    }

    pub fn find_pointer(&self, pointer: Pointer) -> Option<Type> {
        self.pointers
            .iter()
            .position(|value| *value == pointer)
            .map(Type::Pointer)
    }

    pub fn pointer(&self, ty: Type) -> Option<Pointer> {
        match ty {
            Type::Pointer(id) => self.pointers.get(id).copied(),
            _ => None,
        }
    }

    /// Whether `a` and `b` hold the same values: equal, or pointers that differ only in
    /// mutability, at any depth. A pointer cast between such pointees is a view.
    pub fn same_values(&self, a: Type, b: Type) -> bool {
        a == b
            || match (self.pointer(a), self.pointer(b)) {
                (Some(a), Some(b)) => {
                    a.space == b.space && a.slice == b.slice && self.same_values(a.target, b.target)
                }
                _ => false,
            }
    }

    pub fn pointers(&self) -> impl Iterator<Item = (usize, &Pointer)> {
        self.pointers.iter().enumerate()
    }

    /// Values that contain no pointer, so they can live in arrays, slices and buffers.
    pub fn is_owned(&self, ty: Type) -> bool {
        match ty {
            Type::Unit | Type::Scalar(_) | Type::Dim3 | Type::Checked(_) | Type::Matrix(_) => true,
            Type::Aggregate(id) => self
                .get(id)
                .is_some_and(|aggregate| aggregate.component_types().all(|t| self.is_owned(t))),
            _ => false,
        }
    }

    pub fn field(&self, ty: Type, index: u32) -> Option<Type> {
        match ty {
            Type::Aggregate(id) => self.get(id)?.field(index),
            Type::Dim3 if index < 3 => Some(Type::Scalar(Scalar::U32)),
            Type::Checked(s) if index == 0 => Some(Type::Scalar(s)),
            Type::Checked(_) if index == 1 => Some(Type::Scalar(Scalar::Bool)),
            _ => None,
        }
    }

    pub fn field_count(&self, ty: Type) -> usize {
        match ty {
            Type::Aggregate(id) => self.get(id).map_or(0, Aggregate::len),
            Type::Dim3 => 3,
            Type::Checked(_) => 2,
            _ => 0,
        }
    }

    pub fn get(&self, id: usize) -> Option<&Aggregate> {
        self.aggregates.get(id)
    }

    /// Owned types whose scalar leaves have a canonical host/GPU representation.
    pub fn is_abi_value(&self, ty: Type) -> bool {
        match ty {
            Type::Scalar(scalar) => !matches!(scalar, Scalar::Bool | Scalar::Usize | Scalar::Isize),
            Type::Aggregate(id) => self.get(id).is_some_and(|aggregate| {
                !aggregate.is_empty()
                    && !matches!(aggregate, Aggregate::Enum { .. })
                    && aggregate.component_types().all(|child| {
                        (!matches!(child, Type::Aggregate(child_id) if child_id >= id))
                            && self.is_abi_value(child)
                    })
            }),
            _ => false,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &Aggregate)> {
        self.aggregates.iter().enumerate()
    }

    pub(crate) fn validate(&self, source: &SourceLocation) -> Result<(), Error> {
        for (id, aggregate) in self.iter() {
            if aggregate.is_empty() {
                return Err(Error::new(source, "empty aggregate types are unsupported"));
            }
            for ty in aggregate.component_types() {
                match ty {
                    Type::Scalar(_) | Type::Dim3 => {}
                    // Zero-sized fields take no storage; array elements need some.
                    Type::Unit if !matches!(aggregate, Aggregate::Array { .. }) => {}
                    Type::Checked(scalar) if scalar.is_integer() || scalar == Scalar::F32 => {}
                    Type::Matrix(scalar) if crate::is_matrix_element(scalar) => {}
                    Type::Aggregate(child) if child < id => {}
                    Type::Aggregate(_) => {
                        return Err(Error::new(
                            source,
                            format!("aggregate type {id} must reference earlier definitions"),
                        ));
                    }
                    Type::Pointer(pointer) if pointer < self.pointers.len() => {}
                    _ => {
                        return Err(Error::new(
                            source,
                            format!("aggregate type {id} requires value components"),
                        ));
                    }
                }
            }
        }
        for pointer in &self.pointers {
            let valid = match pointer.target {
                Type::Aggregate(id) => id < self.aggregates.len(),
                Type::Pointer(target) => target < self.pointers.len(),
                target => !matches!(target, Type::Unit | Type::Never | Type::Buffer { .. }),
            };
            if !valid {
                return Err(Error::new(source, "pointers must target values"));
            }
        }
        Ok(())
    }
}
