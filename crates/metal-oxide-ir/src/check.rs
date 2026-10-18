use crate::OverflowOp;

/// The Rust panic a failed assertion reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    Bounds,
    Overflow(OverflowOp),
    DivisionByZero,
    RemainderByZero,
    /// `slice[start..]` with `start > len`.
    SliceStart,
    /// `slice[..end]` with `end > len`.
    SliceEnd,
    /// `slice[start..end]` with `start > end`.
    SliceOrder,
}
