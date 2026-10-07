use crate::{BinaryOp, Operand, Place};

use super::{domain::Range, state::State};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Atom {
    Place(Place),
    Integer(i128),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Comparison {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Predicate {
    Compare(Comparison, Atom, Atom),
    Not(Box<Self>),
    And(Vec<Self>),
    Or(Vec<Self>),
}

impl Predicate {
    pub fn uses(&self, destination: &Place) -> bool {
        match self {
            Self::Compare(_, a, b) => [a, b].iter().any(|a| matches!(a, Atom::Place(p) if p.contains(destination) || destination.contains(p))),
            Self::Not(p) => p.uses(destination),
            Self::And(values) | Self::Or(values) => values.iter().any(|p| p.uses(destination)),
        }
    }

    pub fn nodes(&self) -> usize {
        match self {
            Self::Compare(..) => 1,
            Self::Not(p) => 1 + p.nodes(),
            Self::And(values) | Self::Or(values) => {
                1 + values.iter().map(Self::nodes).sum::<usize>()
            }
        }
    }

    pub fn negate(self) -> Self {
        match self {
            Self::Not(p) => *p,
            _ => Self::Not(Box::new(self)),
        }
    }

    pub fn combine(a: Self, b: Self, and: bool) -> Self {
        let mut values = Vec::new();
        for p in [a, b] {
            match p {
                Self::And(parts) if and => values.extend(parts),
                Self::Or(parts) if !and => values.extend(parts),
                _ => values.push(p),
            }
        }
        values.sort();
        values.dedup();
        if and {
            Self::And(values)
        } else {
            Self::Or(values)
        }
    }

    pub fn value(&self, state: &State) -> Range {
        if let Some(value) = state.truths.get(self) {
            return Range::exact(i128::from(*value));
        }
        match self {
            Self::Compare(op, a, b) if a == b => Range::exact(i128::from(matches!(
                op,
                Comparison::Eq | Comparison::Le | Comparison::Ge
            ))),
            Self::Compare(op, a, b) => compare(*op, &state.atom(a), &state.atom(b)),
            Self::Not(p) => {
                let value = p.value(state);
                Range::new(1 - value.hi, 1 - value.lo)
            }
            Self::And(parts) => {
                if parts.iter().any(|p| p.value(state).hi == 0) {
                    Range::exact(0)
                } else if parts.iter().all(|p| p.value(state).lo == 1) {
                    Range::exact(1)
                } else {
                    Range::new(0, 1)
                }
            }
            Self::Or(parts) => {
                if parts.iter().any(|p| p.value(state).lo == 1) {
                    Range::exact(1)
                } else if parts.iter().all(|p| p.value(state).hi == 0) {
                    Range::exact(0)
                } else {
                    Range::new(0, 1)
                }
            }
        }
    }
}

pub(super) fn comparison(op: BinaryOp) -> Option<Comparison> {
    use BinaryOp::*;
    Some(match op {
        Eq => Comparison::Eq,
        Ne => Comparison::Ne,
        Lt => Comparison::Lt,
        Le => Comparison::Le,
        Gt => Comparison::Gt,
        Ge => Comparison::Ge,
        _ => return None,
    })
}

pub(super) fn compare(op: Comparison, a: &Range, b: &Range) -> Range {
    use Comparison::*;
    let true_value = match op {
        Eq => a.singleton().is_some() && a.singleton() == b.singleton(),
        Ne => {
            a.hi < b.lo
                || b.hi < a.lo
                || b.singleton().is_some_and(|n| !a.contains(n))
                || a.singleton().is_some_and(|n| !b.contains(n))
        }
        Lt => a.hi < b.lo,
        Le => a.hi <= b.lo,
        Gt => a.lo > b.hi,
        Ge => a.lo >= b.hi,
    };
    let false_value = match op {
        Eq => {
            a.hi < b.lo
                || b.hi < a.lo
                || b.singleton().is_some_and(|n| !a.contains(n))
                || a.singleton().is_some_and(|n| !b.contains(n))
        }
        Ne => a.singleton().is_some() && a.singleton() == b.singleton(),
        Lt => a.lo >= b.hi,
        Le => a.lo > b.hi,
        Gt => a.hi <= b.lo,
        Ge => a.hi < b.lo,
    };
    if true_value {
        Range::exact(1)
    } else if false_value {
        Range::exact(0)
    } else {
        Range::new(0, 1)
    }
}

pub(super) fn from_binary(
    state: &State,
    op: BinaryOp,
    a: &Operand,
    b: &Operand,
) -> Option<Predicate> {
    let left = state.predicate(a);
    let right = state.predicate(b);
    if matches!(op, BinaryOp::BitAnd | BinaryOp::BitOr) {
        return Some(Predicate::combine(left?, right?, op == BinaryOp::BitAnd));
    }
    if matches!(op, BinaryOp::Eq | BinaryOp::Ne) {
        let (predicate, value) = if let Some(p) = left {
            (p, state.constant(b)?)
        } else if let Some(p) = right {
            (p, state.constant(a)?)
        } else {
            return atoms(state, op, a, b);
        };
        if value == 0 || value == 1 {
            return Some(if (value == 1) == (op == BinaryOp::Eq) {
                predicate
            } else {
                predicate.negate()
            });
        }
    }
    atoms(state, op, a, b)
}

fn atoms(state: &State, op: BinaryOp, a: &Operand, b: &Operand) -> Option<Predicate> {
    let op = comparison(op)?;
    let mut a = state.symbol(a)?;
    let mut b = state.symbol(b)?;
    if matches!(op, Comparison::Eq | Comparison::Ne) && b < a {
        std::mem::swap(&mut a, &mut b);
    }
    Some(Predicate::Compare(op, a, b))
}
