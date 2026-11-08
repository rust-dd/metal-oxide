use std::fmt;

use crate::{Access, Layout, LayoutKind, ParameterType, Scalar};

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::I8 => "i8",
            Self::I16 => "i16",
        })
    }
}

/// Names the Rust type: `f32`, `[u32; 4]`, `(f32, u32)` or a record's name.
impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            LayoutKind::Scalar { scalar } => scalar.fmt(f),
            LayoutKind::Array {
                element, length, ..
            } => write!(f, "[{element}; {length}]"),
            LayoutKind::Tuple { fields } => {
                f.write_str("(")?;
                for (index, field) in fields.iter().enumerate() {
                    let separator = if index == 0 { "" } else { ", " };
                    write!(f, "{separator}{}", field.layout)?;
                }
                f.write_str(if fields.len() == 1 { ",)" } else { ")" })
            }
            LayoutKind::Record { name, .. } => f.write_str(name),
        }
    }
}

impl fmt::Display for Access {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::ReadWrite => "read-write",
            Self::Atomic => "atomic",
        })
    }
}

/// Describes the argument a parameter takes, such as `read buffer of f32`.
impl fmt::Display for ParameterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value { layout } => write!(f, "value of {layout}"),
            Self::Buffer {
                element, access, ..
            } => write!(f, "{access} buffer of {element}"),
            Self::Threadgroup { element, .. } => write!(f, "threadgroup buffer of {element}"),
        }
    }
}

#[cfg(test)]
mod tests;
