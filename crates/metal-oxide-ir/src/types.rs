use crate::{Error, SourceLocation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    Bool,
    F32,
    F16,
    U32,
    I32,
    U8,
    U16,
    I8,
    I16,
    /// Device pointer-width integer, restricted to internal values.
    Usize,
}

impl Scalar {
    pub const fn bits(self) -> u32 {
        match self {
            Self::U8 | Self::I8 => 8,
            Self::U16 | Self::I16 | Self::F16 => 16,
            Self::Bool => 1,
            Self::F32 | Self::U32 | Self::I32 => 32,
            Self::Usize => 64,
        }
    }

    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::U8 | Self::U16 | Self::U32 | Self::I32 | Self::I8 | Self::I16 | Self::Usize
        )
    }

    pub const fn is_signed(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::Usize => "usize",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
    Atomic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressSpace {
    Device,
    Threadgroup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    pub fn len(&self) -> usize {
        match self {
            Self::Record { fields, .. } => fields.len(),
            Self::Tuple(fields) => fields.len(),
            Self::Array { length, .. } => *length as usize,
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
        }
    }

    /// Direct component types; an array contributes its element type once.
    pub fn component_types(&self) -> impl Iterator<Item = Type> + '_ {
        let count = if matches!(self, Self::Array { .. }) {
            1
        } else {
            self.len()
        };
        (0..count).map(|index| match self {
            Self::Array { element, .. } => *element,
            _ => self.field(index as u32).unwrap(),
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TypeTable {
    aggregates: Vec<Aggregate>,
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
            Type::Scalar(scalar) => !matches!(scalar, Scalar::Bool | Scalar::Usize),
            Type::Aggregate(id) => self.get(id).is_some_and(|aggregate| {
                !aggregate.is_empty()
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
                    Type::Checked(scalar) if scalar.is_integer() => {}
                    Type::Aggregate(child) if child < id => {}
                    Type::Aggregate(_) => {
                        return Err(Error::new(
                            source,
                            format!("aggregate type {id} must reference earlier definitions"),
                        ));
                    }
                    _ => {
                        return Err(Error::new(
                            source,
                            format!("aggregate type {id} requires owned value components"),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}
