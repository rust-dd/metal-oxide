use serde::{Deserialize, Serialize};

/// A runtime check in kernel code; its index in the manifest is the check id.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Check {
    pub kind: CheckKind,
    pub location: Location,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum CheckKind {
    Bounds,
    Overflow(OverflowOp),
    DivisionByZero,
    RemainderByZero,
    Panic(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverflowOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    Shl,
    Shr,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Location {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

impl CheckKind {
    /// Rust's runtime panic message; bounds failures do not record the length and index.
    pub fn message(&self) -> String {
        match self {
            Self::Bounds => "index out of bounds".into(),
            Self::Overflow(op) => {
                let operation = match op {
                    OverflowOp::Add => "add",
                    OverflowOp::Sub => "subtract",
                    OverflowOp::Mul => "multiply",
                    OverflowOp::Div => "divide",
                    OverflowOp::Rem => "calculate the remainder",
                    OverflowOp::Neg => "negate",
                    OverflowOp::Shl => "shift left",
                    OverflowOp::Shr => "shift right",
                };
                format!("attempt to {operation} with overflow")
            }
            Self::DivisionByZero => "attempt to divide by zero".into(),
            Self::RemainderByZero => {
                "attempt to calculate the remainder with a divisor of zero".into()
            }
            Self::Panic(message) => message.clone(),
        }
    }
}
