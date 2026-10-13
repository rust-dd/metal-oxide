use std::collections::BTreeMap;

use super::{
    domain::{Range, integer},
    predicate::{Atom, Comparison, compare},
    state::State,
};
use crate::*;

/// Relations between places as the code names them. Unlike facts and symbols, they never fold a
/// place's current value into a constant, so loop iterations agree on them and joins keep them.
impl State<'_> {
    /// Records that `destination` copies `source`, field by field for small aggregates. Keeping
    /// only the direct source makes loop iterations record alike; queries follow the chain.
    pub fn copy(&mut self, destination: &Place, source: &Place) {
        let (Some(destination), Some(source)) = (self.resolve(destination), self.resolve(source))
        else {
            return;
        };
        if destination.contains(&source) || source.contains(&destination) {
            return;
        }
        let fields = place_type(
            self.module,
            self.function,
            &destination,
            &self.function.source,
        )
        .map_or(0, |ty| self.module.types.field_count(ty));
        if (1..=16).contains(&fields) {
            for field in 0..fields as u32 {
                let mut to = destination.clone();
                to.projection.push(Projection::Field(field));
                let mut from = source.clone();
                from.projection.push(Projection::Field(field));
                self.copy(&to, &from);
            }
            return;
        }
        if let Some(length @ Atom::Length(_)) = self.aliases.get(&source).cloned() {
            self.aliases.insert(destination.clone(), length);
        }
        self.copies.insert(destination, source);
    }

    /// The oldest place whose unchanged value `place` holds.
    fn root(&self, place: &Place) -> Place {
        let mut root = place;
        while let Some(source) = self.copies.get(root) {
            root = source;
        }
        root.clone()
    }

    /// Whether two atoms name the same value.
    fn same(&self, a: &Atom, b: &Atom) -> bool {
        match (a, b) {
            (Atom::Place(a), Atom::Place(b)) => self.root(a) == self.root(b),
            _ => a == b,
        }
    }

    /// Whether the atoms' ranges, an assumed relation, or a stronger one give `a op b`.
    fn holds(&self, a: &Atom, op: Comparison, b: &Atom) -> bool {
        if compare(op, &self.atom(a), &self.atom(b)).singleton() == Some(1) {
            return true;
        }
        let stronger = |relation: Comparison| {
            relation == op
                || matches!(
                    (op, relation),
                    (Comparison::Le, Comparison::Lt)
                        | (Comparison::Ge, Comparison::Gt)
                        | (Comparison::Ne, Comparison::Lt | Comparison::Gt)
                )
        };
        self.relations
            .iter()
            .any(|(x, relation, y)| stronger(*relation) && self.same(x, a) && self.same(y, b))
    }

    /// Before `destination` is overwritten, a copy of its old value takes over the relations
    /// and copies that named it.
    pub(super) fn inherit(&mut self, destination: &Place) {
        let heirs = self
            .copies
            .iter()
            .filter(|(copy, source)| {
                destination.contains(source)
                    && !destination.contains(copy)
                    && !copy.contains(destination)
            })
            .map(|(copy, source)| (source.clone(), copy.clone()))
            .collect::<BTreeMap<_, _>>();
        if heirs.is_empty() {
            return;
        }
        let rename = |atom: &Atom| match atom {
            Atom::Place(place) => heirs
                .get(place)
                .map_or_else(|| atom.clone(), |heir| Atom::Place(heir.clone())),
            _ => atom.clone(),
        };
        self.relations = self
            .relations
            .iter()
            .map(|(a, op, b)| (rename(a), *op, rename(b)))
            .collect();
        for (_, a, b) in self.conditions.values_mut() {
            (*a, *b) = (rename(a), rename(b));
        }
        let copies = std::mem::take(&mut self.copies);
        self.copies = copies
            .into_iter()
            .filter(|(copy, _)| !heirs.values().any(|heir| heir == copy))
            .map(|(copy, source)| match heirs.get(&source) {
                Some(heir) => (copy, heir.clone()),
                None => (copy, source),
            })
            .collect();
    }

    /// An operand as the code names it: a place, a slice length, or a literal. Unlike symbols,
    /// written atoms never fold a place's current value into a constant.
    pub fn written(&self, operand: &Operand) -> Option<Atom> {
        match operand {
            Operand::Place(place) => {
                let place = self.resolve(place)?;
                let length = [&place, &self.root(&place)]
                    .into_iter()
                    .find_map(|place| match self.aliases.get(place) {
                        Some(length @ Atom::Length(_)) => Some(length.clone()),
                        _ => None,
                    });
                Some(length.unwrap_or(Atom::Place(place)))
            }
            Operand::Constant(constant) => integer(*constant).map(Atom::Integer),
            Operand::AggregateConstant { .. } => None,
        }
    }

    /// Records the comparison a boolean place holds after `destination = a op b`.
    pub fn record_comparison(
        &mut self,
        destination: &Place,
        op: Comparison,
        a: &Operand,
        b: &Operand,
    ) {
        if let (Some(destination), Some(a), Some(b)) =
            (self.resolve(destination), self.written(a), self.written(b))
        {
            self.conditions.insert(destination, (op, a, b));
        }
    }

    /// The slice place a reborrow views whole, or the place itself.
    fn slice(&self, place: &Place) -> Option<Place> {
        let place = self.resolve(place)?;
        Some(self.slices.get(&place).cloned().unwrap_or(place))
    }

    /// Records that `destination` reborrows all of `slice`.
    pub fn reborrow(&mut self, destination: &Place, slice: &Place) {
        if let (Some(destination), Some(slice)) = (self.resolve(destination), self.slice(slice)) {
            if let Some(length) = self.lengths.get(&slice).cloned() {
                self.lengths.insert(destination.clone(), length);
            }
            self.slices.insert(destination, slice);
        }
    }

    /// Records that `destination` slices a whole array of `length` elements.
    pub fn extent(&mut self, destination: &Place, length: u32) {
        if let Some(destination) = self.resolve(destination) {
            self.lengths
                .insert(destination, Range::exact(i128::from(length)));
        }
    }

    /// The bounds known for a slice's length.
    pub fn length_range(&self, slice: &Place) -> Option<Range> {
        let slice = self.slice(slice)?;
        Some(self.atom(&Atom::Length(slice)))
    }

    /// Records that `destination` holds the length of `slice`, a symbol that outlives it.
    pub fn length(&mut self, destination: &Place, slice: &Place) {
        if let (Some(destination), Some(slice)) = (self.resolve(destination), self.slice(slice)) {
            self.aliases.insert(destination, Atom::Length(slice));
        }
    }

    /// Whether `slice`'s length bounds `atom` with `op`, by an assumed comparison or because
    /// `atom` is the length itself.
    pub fn bounded(&self, atom: &Atom, op: Comparison, slice: &Place) -> bool {
        let Some(slice) = self.slice(slice) else {
            return false;
        };
        let length = Atom::Length(slice);
        (op == Comparison::Le && *atom == length) || self.holds(atom, op, &length)
    }

    /// Whether `start <= end` holds for a subslice.
    pub fn ordered(&self, start: &Atom, end: &Atom) -> bool {
        *start == Atom::Integer(0)
            || self.same(start, end)
            || self.holds(start, Comparison::Le, end)
    }

    /// The atoms an index local can match: its place and its constant value.
    pub fn index_atoms(&self, index: usize) -> Vec<Atom> {
        let operand = Operand::local(index);
        let mut atoms = Vec::from_iter(self.written(&operand));
        atoms.extend(self.constant(&operand).map(Atom::Integer));
        atoms
    }

    /// Whether an assumed comparison between the same written atoms decides the condition.
    pub fn implied(&self, operand: &Operand, expected: bool) -> bool {
        let Some((op, a, b)) = self
            .resolve_operand(operand)
            .and_then(|place| self.conditions.get(&place))
        else {
            return false;
        };
        let op = if expected { *op } else { op.negate() };
        self.holds(a, op, b)
    }

    fn resolve_operand(&self, operand: &Operand) -> Option<Place> {
        match operand {
            Operand::Place(place) => self.resolve(place),
            _ => None,
        }
    }
}
