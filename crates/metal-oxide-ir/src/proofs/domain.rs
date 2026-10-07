use std::collections::BTreeSet;

use crate::{BinaryOp, Constant, Scalar};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Range {
    pub lo: i128,
    pub hi: i128,
    pub excluded: BTreeSet<i128>,
}

impl Range {
    pub fn new(lo: i128, hi: i128) -> Self {
        Self {
            lo,
            hi,
            excluded: BTreeSet::new(),
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
        if self.excluded.len() < 8 || value == self.lo || value == self.hi {
            self.excluded.insert(value);
        }
        self.restrict(self.lo, self.hi)
    }

    pub fn join(&self, other: &Self) -> Self {
        let mut result = Self::new(self.lo.min(other.lo), self.hi.max(other.hi));
        result.excluded = self
            .excluded
            .union(&other.excluded)
            .copied()
            .filter(|v| !self.contains(*v) && !other.contains(*v))
            .take(8)
            .collect();
        result
    }

    pub fn widen(&self, next: &Self, scalar: Scalar) -> Self {
        let full = Self::full(scalar);
        Self::new(
            if next.lo < self.lo { full.lo } else { next.lo },
            if next.hi > self.hi { full.hi } else { next.hi },
        )
    }
}

pub(super) fn integer(value: Constant) -> Option<i128> {
    Some(match value {
        Constant::Bool(v) => i128::from(v),
        Constant::U8(v) => i128::from(v),
        Constant::U16(v) => i128::from(v),
        Constant::U32(v) => i128::from(v),
        Constant::Usize(v) => i128::from(v),
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
