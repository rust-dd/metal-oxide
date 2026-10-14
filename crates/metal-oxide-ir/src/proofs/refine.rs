use crate::{Operand, Place};

use super::{
    domain::Range,
    predicate::{Atom, Comparison, Predicate},
    state::State,
};

impl State<'_> {
    /// Assumes a kernel requirement that compares parameters' lengths, integer values and
    /// literals; a launch checks it before encoding. Arithmetic terms stay runtime checks.
    pub fn require(&mut self, contract: &metal_oxide_artifact::Contract) {
        use metal_oxide_artifact::{Comparison as Relation, Term};
        let atom = |term: &Term| match term {
            Term::Length(index) => Some(Atom::Length(Place::local(index + 1))),
            Term::Value(index) => Some(Atom::Place(Place::local(index + 1))),
            Term::Literal(value) => Some(Atom::Integer(i128::from(*value))),
            Term::Add(..) | Term::Sub(..) | Term::Mul(..) => None,
        };
        let (Some(a), Some(b)) = (atom(&contract.left), atom(&contract.right)) else {
            return;
        };
        let op = match contract.comparison {
            Relation::Eq => Comparison::Eq,
            Relation::Ne => Comparison::Ne,
            Relation::Lt => Comparison::Lt,
            Relation::Le => Comparison::Le,
            Relation::Gt => Comparison::Gt,
            Relation::Ge => Comparison::Ge,
        };
        self.refine(&Predicate::Compare(op, a.clone(), b.clone()), true);
        self.relations.insert((a, op, b));
    }

    pub fn assume(&mut self, operand: &Operand, expected: bool) -> bool {
        if let Some(value) = self.operand_range(operand)
            && !value.contains(i128::from(expected))
        {
            return false;
        }
        if let Some(predicate) = self.predicate(operand)
            && !self.refine(&predicate, expected)
        {
            return false;
        }
        if let Operand::Place(p) = operand
            && let Some(p) = self.resolve(p)
        {
            if let Some((op, a, b)) = self.conditions.get(&p).cloned() {
                let op = if expected { op } else { op.negate() };
                self.relations.insert((a, op, b));
            }
            self.facts.insert(p, Range::exact(i128::from(expected)));
        }
        true
    }

    pub fn refine(&mut self, predicate: &Predicate, expected: bool) -> bool {
        if !predicate.value(self).contains(i128::from(expected)) {
            return false;
        }
        let valid = match predicate {
            Predicate::Not(p) => self.refine(p, !expected),
            Predicate::And(values) if expected => values.iter().all(|p| self.refine(p, true)),
            Predicate::Or(values) if !expected => values.iter().all(|p| self.refine(p, false)),
            Predicate::And(values) | Predicate::Or(values) => {
                if values
                    .iter()
                    .any(|p| p.value(self).singleton() == Some(i128::from(expected)))
                {
                    return true;
                }
                let relevant = values
                    .iter()
                    .filter(|p| p.value(self).singleton().is_none())
                    .cloned()
                    .collect::<Vec<_>>();
                relevant.len() != 1 || self.refine(&relevant[0], expected)
            }
            Predicate::Compare(op, a, b) => self.refine_comparison(*op, a, b, expected),
        };
        if valid {
            self.truths.insert(predicate.clone(), expected);
        }
        valid
    }

    fn refine_comparison(&mut self, op: Comparison, a: &Atom, b: &Atom, expected: bool) -> bool {
        use Comparison::*;
        let op = if expected { op } else { op.negate() };
        let left = self.atom(a);
        let right = self.atom(b);
        let mut x = left.clone();
        let mut y = right.clone();
        let valid = match op {
            Eq => x.restrict(right.lo, right.hi) && y.restrict(left.lo, left.hi),
            Ne => {
                right.singleton().is_none_or(|n| x.exclude(n))
                    && left.singleton().is_none_or(|n| y.exclude(n))
            }
            Lt => x.restrict(left.lo, right.hi - 1) && y.restrict(left.lo + 1, right.hi),
            Le => x.restrict(left.lo, right.hi) && y.restrict(left.lo, right.hi),
            Gt => x.restrict(right.lo + 1, left.hi) && y.restrict(right.lo, left.hi - 1),
            Ge => x.restrict(right.lo, left.hi) && y.restrict(right.lo, left.hi),
        };
        if !valid {
            return false;
        }
        for (atom, range) in [(a, x), (b, y)] {
            match atom {
                Atom::Place(p) => {
                    self.facts.insert(p.clone(), range);
                }
                Atom::Length(slice) => {
                    self.lengths.insert(slice.clone(), range);
                }
                Atom::Integer(_) => {}
            }
        }
        true
    }

    pub fn condition(&self, operand: &Operand) -> Option<bool> {
        let range = self
            .predicate(operand)
            .map(|p| p.value(self))
            .or_else(|| self.operand_range(operand))?;
        range.singleton().map(|n| n != 0)
    }

    pub fn restrict_place(&mut self, place: &Place, value: i128, equal: bool) -> bool {
        let mut range = self.range(place).unwrap();
        let valid = if equal {
            range.restrict(value, value)
        } else {
            range.exclude(value)
        };
        if valid {
            self.facts.insert(place.clone(), range);
        }
        valid
    }
}
