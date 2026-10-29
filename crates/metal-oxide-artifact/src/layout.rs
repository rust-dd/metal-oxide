use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{Error, Scalar, validate::identifier};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layout {
    pub size: u64,
    pub alignment: u64,
    pub kind: LayoutKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LayoutKind {
    Scalar {
        scalar: Scalar,
    },
    Array {
        element: Box<Layout>,
        length: u32,
        stride: u64,
    },
    Tuple {
        fields: Vec<FieldLayout>,
    },
    Record {
        name: String,
        fields: Vec<FieldLayout>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldLayout {
    pub name: String,
    pub offset: u64,
    pub layout: Layout,
}

impl Layout {
    pub fn scalar(scalar: Scalar) -> Self {
        Self {
            size: scalar.size() as u64,
            alignment: scalar.alignment() as u64,
            kind: LayoutKind::Scalar { scalar },
        }
    }

    pub fn scalar_type(&self) -> Option<Scalar> {
        match self.kind {
            LayoutKind::Scalar { scalar } => Some(scalar),
            _ => None,
        }
    }

    pub fn array(element: Self, length: u32) -> Result<Self, Error> {
        element.validate()?;
        if length == 0 {
            return Err(Error("zero-length ABI arrays are unsupported".into()));
        }
        let size = element
            .size
            .checked_mul(u64::from(length))
            .ok_or_else(overflow)?;
        Ok(Self {
            size,
            alignment: element.alignment,
            kind: LayoutKind::Array {
                stride: element.size,
                element: Box::new(element),
                length,
            },
        })
    }

    pub fn tuple(elements: Vec<Self>) -> Result<Self, Error> {
        let values = elements
            .into_iter()
            .enumerate()
            .map(|(index, layout)| (format!("f{index}"), layout))
            .collect::<Vec<_>>();
        let (size, alignment, fields) = fields(values)?;
        Ok(Self {
            size,
            alignment,
            kind: LayoutKind::Tuple { fields },
        })
    }

    pub fn record(name: impl Into<String>, values: Vec<(String, Self)>) -> Result<Self, Error> {
        let name = name.into();
        if !identifier(&name) || reserved(&name) {
            return Err(Error("invalid ABI record name".into()));
        }
        let (size, alignment, fields) = fields(values)?;
        Ok(Self {
            size,
            alignment,
            kind: LayoutKind::Record { name, fields },
        })
    }

    /// Checks the canonical size, alignment, offsets, stride and names without rebuilding the layout.
    pub fn validate(&self) -> Result<(), Error> {
        let canonical = match &self.kind {
            LayoutKind::Scalar { scalar } => {
                self.size == scalar.size() as u64 && self.alignment == scalar.alignment() as u64
            }
            LayoutKind::Array {
                element,
                length,
                stride,
            } => {
                element.validate()?;
                if *length == 0 {
                    return Err(Error("zero-length ABI arrays are unsupported".into()));
                }
                let size = element
                    .size
                    .checked_mul(u64::from(*length))
                    .ok_or_else(overflow)?;
                *stride == element.size && self.size == size && self.alignment == element.alignment
            }
            LayoutKind::Tuple { fields } => {
                let placed = placed(self, fields, false)?;
                placed
                    && fields
                        .iter()
                        .enumerate()
                        .all(|(index, field)| field.name == format!("f{index}"))
            }
            LayoutKind::Record { name, fields } => {
                if !identifier(name) || reserved(name) {
                    return Err(Error("invalid ABI record name".into()));
                }
                placed(self, fields, true)?
            }
        };
        if !canonical {
            return Err(Error(
                "layout differs from its canonical size, alignment, offsets, or stride".into(),
            ));
        }
        Ok(())
    }
}

fn reserved(name: &str) -> bool {
    matches!(name, "self" | "Self" | "super" | "crate" | "_")
}

/// Whether the fields sit at their canonical offsets inside the layout's size and alignment.
fn placed(layout: &Layout, fields: &[FieldLayout], named: bool) -> Result<bool, Error> {
    if fields.is_empty() {
        return Err(Error("empty ABI aggregates are unsupported".into()));
    }
    let mut names = HashSet::with_capacity(fields.len());
    let mut offsets = true;
    let mut size = 0;
    let mut alignment = 1;
    for field in fields {
        if named
            && (!identifier(&field.name)
                || reserved(&field.name)
                || !names.insert(field.name.as_str()))
        {
            return Err(Error("invalid or duplicate ABI field name".into()));
        }
        field.layout.validate()?;
        let offset = aligned(size, field.layout.alignment)?;
        offsets &= field.offset == offset;
        size = offset.checked_add(field.layout.size).ok_or_else(overflow)?;
        alignment = alignment.max(field.layout.alignment);
    }
    Ok(offsets && layout.size == aligned(size, alignment)? && layout.alignment == alignment)
}

fn overflow() -> Error {
    Error("ABI layout size overflow".into())
}

fn aligned(value: u64, alignment: u64) -> Result<u64, Error> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(overflow)
}

fn fields(values: Vec<(String, Layout)>) -> Result<(u64, u64, Vec<FieldLayout>), Error> {
    if values.is_empty() {
        return Err(Error("empty ABI aggregates are unsupported".into()));
    }
    let mut names = HashSet::new();
    let mut result = Vec::with_capacity(values.len());
    let mut size = 0;
    let mut alignment = 1;
    for (name, layout) in values {
        if !identifier(&name) || reserved(&name) || !names.insert(name.clone()) {
            return Err(Error("invalid or duplicate ABI field name".into()));
        }
        layout.validate()?;
        let offset = aligned(size, layout.alignment)?;
        size = offset.checked_add(layout.size).ok_or_else(overflow)?;
        alignment = alignment.max(layout.alignment);
        result.push(FieldLayout {
            name,
            offset,
            layout,
        });
    }
    Ok((aligned(size, alignment)?, alignment, result))
}
