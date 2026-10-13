/// The Rust panic a failed assertion reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    Bounds,
    Overflow(OverflowOp),
    DivisionByZero,
    RemainderByZero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
