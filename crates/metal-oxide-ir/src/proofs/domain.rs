use std::collections::BTreeSet;

use crate::{BinaryOp, Builtin, Constant, Scalar};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Range {
    pub lo: i128,
    pub hi: i128,
    /// Sorted, without duplicates.
    pub excluded: Vec<i128>,
    pub origin: Option<Origin>,
}

/// The launch quantity a value equals, which bounds products that intervals cannot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Origin {
    Coordinate(Builtin, u32),
    /// `block_idx * block_dim` on one axis.
    BlockStart(u32),
}

impl Range {
    pub fn new(lo: i128, hi: i128) -> Self {
        Self {
            lo,
            hi,
            excluded: Vec::new(),
            origin: None,
        }
    }

    /// One axis of a launch coordinate. Launch validation keeps block and grid sizes positive,
    /// their product within u32, and a kernel's required block size exact.
    pub fn coordinate(builtin: Builtin, axis: u32, block: Option<u32>) -> Self {
        let limit = i128::from(u32::MAX);
        let block = block.map(i128::from);
        let (lo, hi) = match builtin {
            Builtin::BlockDim => block.map_or((1, limit), |block| (block, block)),
            Builtin::ThreadIdx => (0, block.map_or(limit - 1, |block| block - 1)),
            Builtin::GridDim => (1, limit / block.unwrap_or(1)),
            Builtin::BlockIdx => (0, limit / block.unwrap_or(1) - 1),
        };
        Self {
            origin: Some(Origin::Coordinate(builtin, axis)),
            ..Self::new(lo, hi)
        }
    }

    pub fn full(scalar: Scalar) -> Self {
        if scalar.is_signed() {
            let limit = 1_i128 << (scalar.bits() - 1);
            Self::new(-limit, limit - 1)
        } else {
            Self::new(0, (1_i128 << scalar.bits()) - 1)
        }
    }

    pub fn exact(value: i128) -> Self {
        Self::new(value, value)
    }

    pub fn singleton(&self) -> Option<i128> {
        (self.lo == self.hi && !self.excluded.contains(&self.lo)).then_some(self.lo)
    }

    pub fn contains(&self, value: i128) -> bool {
        value >= self.lo && value <= self.hi && !self.excluded.contains(&value)
    }

    pub fn restrict(&mut self, lo: i128, hi: i128) -> bool {
        self.lo = self.lo.max(lo);
        self.hi = self.hi.min(hi);
        while self.excluded.contains(&self.lo) && self.lo <= self.hi {
            self.lo += 1;
        }
        while self.excluded.contains(&self.hi) && self.lo <= self.hi {
            self.hi -= 1;
        }
        self.excluded.retain(|v| *v >= self.lo && *v <= self.hi);
        self.lo <= self.hi
    }

    pub fn exclude(&mut self, value: i128) -> bool {
        if (self.excluded.len() < 8 || value == self.lo || value == self.hi)
            && let Err(position) = self.excluded.binary_search(&value)
        {
            self.excluded.insert(position, value);
        }
        self.restrict(self.lo, self.hi)
    }

    pub fn join(&self, other: &Self) -> Self {
        let mut result = Self::new(self.lo.min(other.lo), self.hi.max(other.hi));
        result.origin = self.origin.filter(|origin| other.origin == Some(*origin));
        result.excluded = self
            .excluded
            .iter()
            .chain(&other.excluded)
            .copied()
            .filter(|v| !self.contains(*v) && !other.contains(*v))
            .collect();
        result.excluded.sort_unstable();
        result.excluded.dedup();
        result.excluded.truncate(8);
        result
    }

    /// Moves a growing bound to the nearest threshold beyond it, or to the type limit.
    pub fn widen(&self, next: &Self, scalar: Scalar, thresholds: &BTreeSet<i128>) -> Self {
        let full = Self::full(scalar);
        Self::new(
            if next.lo < self.lo {
                thresholds
                    .range(full.lo..=next.lo)
                    .next_back()
                    .copied()
                    .unwrap_or(full.lo)
            } else {
                next.lo
            },
            if next.hi > self.hi {
                thresholds
                    .range(next.hi..=full.hi)
                    .next()
                    .copied()
                    .unwrap_or(full.hi)
            } else {
                next.hi
            },
        )
    }
}

pub(super) fn integer(value: Constant) -> Option<i128> {
    Some(match value {
        Constant::Bool(v) => i128::from(v),
        Constant::U8(v) => i128::from(v),
        Constant::U16(v) => i128::from(v),
        Constant::U32(v) => i128::from(v),
        Constant::U64(v) | Constant::Usize(v) => i128::from(v),
        Constant::I8(v) => i128::from(v),
        Constant::I16(v) => i128::from(v),
        Constant::I32(v) => i128::from(v),
        _ => return None,
    })
}

pub(super) fn wrap(value: i128, scalar: Scalar) -> i128 {
    let modulus = 1_i128 << scalar.bits();
    let bits = value.rem_euclid(modulus);
    if scalar.is_signed() && bits >= modulus / 2 {
        bits - modulus
    } else {
        bits
    }
}

pub(super) fn arithmetic(op: BinaryOp, a: &Range, b: &Range, scalar: Scalar) -> (Range, Range) {
    let (value, overflow) = interval(op, a, b, scalar);
    if overflow.singleton() != Some(0)
        && !scalar.is_signed()
        && scalar.bits() >= 32
        && let Some(value) = launch_arithmetic(op, a.origin, b.origin)
    {
        return (value, Range::exact(0));
    }
    (value, overflow)
}

fn interval(op: BinaryOp, a: &Range, b: &Range, scalar: Scalar) -> (Range, Range) {
    use BinaryOp::*;
    let full = Range::full(scalar);
    let bounds = match op {
        Add | AddWithOverflow => a.lo.checked_add(b.lo).zip(a.hi.checked_add(b.hi)),
        Sub | SubWithOverflow => a.lo.checked_sub(b.hi).zip(a.hi.checked_sub(b.lo)),
        Mul | MulWithOverflow => {
            let values = [
                a.lo.checked_mul(b.lo),
                a.lo.checked_mul(b.hi),
                a.hi.checked_mul(b.lo),
                a.hi.checked_mul(b.hi),
            ];
            if values.iter().all(Option::is_some) {
                let values = values.map(Option::unwrap);
                Some((*values.iter().min().unwrap(), *values.iter().max().unwrap()))
            } else {
                None
            }
        }
        BitAnd | BitOr | BitXor | Shl | Shr
            if a.singleton().is_some() && b.singleton().is_some() =>
        {
            let a = a.singleton().unwrap();
            let b = b.singleton().unwrap();
            let shift = b.rem_euclid(i128::from(scalar.bits())) as u32;
            let result = match op {
                BitAnd => a & b,
                BitOr => a | b,
                BitXor => a ^ b,
                Shl => wrap(a << shift, scalar),
                Shr => a >> shift,
                _ => unreachable!(),
            };
            Some((result, result))
        }
        Div | Rem if !b.contains(0) => {
            if let (Some(a), Some(b)) = (a.singleton(), b.singleton()) {
                Some(if op == Div {
                    (a / b, a / b)
                } else {
                    (a % b, a % b)
                })
            } else if op == Rem && b.lo > 0 && a.lo >= 0 {
                Some((0, a.hi.min(b.hi - 1)))
            } else if op == Div {
                let mut negative = b.hi.min(-1);
                while negative >= b.lo && !b.contains(negative) {
                    negative -= 1;
                }
                let mut positive = b.lo.max(1);
                while positive <= b.hi && !b.contains(positive) {
                    positive += 1;
                }
                let divisors = [b.lo, b.hi, negative, positive]
                    .into_iter()
                    .filter(|n| b.contains(*n) && *n != 0);
                let values = divisors
                    .flat_map(|n| [a.lo / n, a.hi / n])
                    .collect::<Vec<_>>();
                Some((*values.iter().min().unwrap(), *values.iter().max().unwrap()))
            } else {
                None
            }
        }
        BitAnd if a.lo >= 0 && b.lo >= 0 => Some((0, a.hi.min(b.hi))),
        _ => None,
    };
    let overflow = match bounds {
        Some((lo, hi)) if lo >= full.lo && hi <= full.hi => Range::exact(0),
        Some((lo, hi)) if hi < full.lo || lo > full.hi => Range::exact(1),
        _ => Range::full(Scalar::Bool),
    };
    let value = match bounds {
        Some((lo, hi)) if lo >= full.lo && hi <= full.hi => Range::new(lo, hi),
        Some((lo, hi)) if lo == hi => Range::exact(wrap(lo, scalar)),
        _ => full,
    };
    (value, overflow)
}

/// Launch validation keeps `grid_dim * block_dim` within u32 on every axis, so a block's start
/// is at most `u32::MAX - block_dim` and a global index at most `u32::MAX - 1`.
fn launch_arithmetic(op: BinaryOp, a: Option<Origin>, b: Option<Origin>) -> Option<Range> {
    use Builtin::*;
    use Origin::*;
    let ordered = |x: Option<Origin>, y: Option<Origin>| Some((x?, y?));
    let pair = ordered(a, b).into_iter().chain(ordered(b, a));
    let limit = i128::from(u32::MAX);
    match op {
        BinaryOp::Mul | BinaryOp::MulWithOverflow => pair.into_iter().find_map(|pair| match pair {
            (Coordinate(BlockIdx, x), Coordinate(BlockDim, y)) if x == y => Some(Range {
                origin: Some(BlockStart(x)),
                ..Range::new(0, limit - 1)
            }),
            (Coordinate(GridDim, x), Coordinate(BlockDim, y)) if x == y => {
                Some(Range::new(1, limit))
            }
            _ => None,
        }),
        BinaryOp::Add | BinaryOp::AddWithOverflow => pair.into_iter().find_map(|pair| match pair {
            (BlockStart(x), Coordinate(ThreadIdx, y)) if x == y => Some(Range::new(0, limit - 1)),
            _ => None,
        }),
        _ => None,
    }
}
