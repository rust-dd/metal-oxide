use crate::{Operand, Place};

use super::{
    domain::Range,
    predicate::{Atom, Comparison, Predicate},
    state::State,
};

impl State<'_> {
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
        let op = if expected {
            op
        } else {
            match op {
                Eq => Ne,
                Ne => Eq,
                Lt => Ge,
                Le => Gt,
                Gt => Le,
                Ge => Lt,
            }
        };
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
        if let Atom::Place(p) = a {
            self.facts.insert(p.clone(), x);
        }
        if let Atom::Place(p) = b {
            self.facts.insert(p.clone(), y);
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
