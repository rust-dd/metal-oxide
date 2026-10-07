use crate::{Error, SourceLocation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    Bool,
    F32,
    U32,
    I32,
    U8,
    U16,
}

impl Scalar {
    pub const fn bits(self) -> u32 {
        match self {
            Self::U8 => 8,
            Self::U16 => 16,
            Self::Bool => 1,
            Self::F32 | Self::U32 | Self::I32 => 32,
        }
    }

    pub const fn is_integer(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::I32)
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
pub enum Type {
    Unit,
    Never,
    Scalar(Scalar),
    Dim3,
    Aggregate(usize),
    Buffer {
        element: Scalar,
        access: Access,
        address_space: AddressSpace,
    },
    /// Integer value and its overflow flag, in that order.
    Checked(Scalar),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Aggregate {
    Record(Vec<Type>),
    Tuple(Vec<Type>),
    Array { element: Type, length: u32 },
}

impl Aggregate {
    pub fn len(&self) -> usize {
        match self {
            Self::Record(fields) | Self::Tuple(fields) => fields.len(),
            Self::Array { length, .. } => *length as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn field(&self, index: u32) -> Option<Type> {
        match self {
            Self::Record(fields) | Self::Tuple(fields) => fields.get(index as usize).copied(),
            Self::Array { element, length } => (index < *length).then_some(*element),
        }
    }

    /// Direct component types; an array contributes its element type once.
    pub fn component_types(&self) -> &[Type] {
        match self {
            Self::Record(fields) | Self::Tuple(fields) => fields,
            Self::Array { element, .. } => std::slice::from_ref(element),
        }
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

    pub fn get(&self, id: usize) -> Option<&Aggregate> {
        self.aggregates.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &Aggregate)> {
        self.aggregates.iter().enumerate()
    }

    pub(crate) fn validate(&self, source: &SourceLocation) -> Result<(), Error> {
        for (id, aggregate) in self.iter() {
            if aggregate.is_empty() {
                return Err(Error::new(source, "empty aggregate types are unsupported"));
            }
            for &ty in aggregate.component_types() {
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
