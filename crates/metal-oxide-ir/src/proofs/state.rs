use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, btree_map},
    iter::Peekable,
};

use super::{
    domain::{Range, integer},
    predicate::{Atom, Comparison, Predicate},
};
use crate::*;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Value(pub BTreeMap<Vec<u32>, Range>);

impl Value {
    pub fn scalar(range: Range) -> Self {
        Self(BTreeMap::from([(Vec::new(), range)]))
    }

    pub fn field(&mut self, index: u32, value: Self) {
        for (path, range) in value.0 {
            self.0
                .insert(std::iter::once(index).chain(path).collect(), range);
        }
    }

    pub fn join(&self, other: &Self) -> Self {
        Self(
            self.0
                .iter()
                .filter_map(|(p, r)| other.0.get(p).map(|s| (p.clone(), r.join(s))))
                .collect(),
        )
    }
}

#[derive(Clone, Debug)]
pub(super) struct State<'a> {
    pub module: &'a Module,
    pub function: &'a Function,
    /// Locals whose address is taken mutably; writes through pointers may change them.
    pub escaped: &'a [usize],
    pub facts: BTreeMap<Place, Range>,
    pub aliases: BTreeMap<Place, Atom>,
    pub predicates: BTreeMap<Place, Predicate>,
    pub truths: BTreeMap<Predicate, bool>,
    /// Reborrowed slices and the slice they view whole.
    pub slices: BTreeMap<Place, Place>,
    /// Places that hold a copy of another place's unchanged value, by that place.
    pub copies: BTreeMap<Place, Place>,
    /// Slices of whole arrays and their lengths.
    pub extents: BTreeMap<Place, u32>,
    /// The comparison each boolean place holds, between places and constants as written.
    pub conditions: BTreeMap<Place, (Comparison, Atom, Atom)>,
    /// Comparisons assumed true. Unlike `truths`, they name places rather than their current
    /// values, so they survive joins of loop iterations.
    pub relations: BTreeSet<(Atom, Comparison, Atom)>,
}

impl PartialEq for State<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.facts == other.facts
            && self.aliases == other.aliases
            && self.predicates == other.predicates
            && self.truths == other.truths
            && self.slices == other.slices
            && self.copies == other.copies
            && self.extents == other.extents
            && self.conditions == other.conditions
            && self.relations == other.relations
    }
}

impl Eq for State<'_> {}

impl<'a> State<'a> {
    pub fn new(module: &'a Module, function: &'a Function, escaped: &'a [usize]) -> Self {
        Self {
            module,
            function,
            escaped,
            facts: BTreeMap::new(),
            aliases: BTreeMap::new(),
            predicates: BTreeMap::new(),
            truths: BTreeMap::new(),
            slices: BTreeMap::new(),
            copies: BTreeMap::new(),
            extents: BTreeMap::new(),
            conditions: BTreeMap::new(),
            relations: BTreeSet::new(),
        }
    }

    pub fn scalar(&self, place: &Place) -> Option<Scalar> {
        match place_type(self.module, self.function, place, &self.function.source).ok()? {
            Type::Scalar(s) if s.is_integer() || s == Scalar::Bool => Some(s),
            _ => None,
        }
    }

    /// The place with constant indices as fields; `None` behind a pointer or a varying index.
    pub fn resolve(&self, place: &Place) -> Option<Place> {
        let mut result = Place::local(place.local);
        for projection in &place.projection {
            result.projection.push(match *projection {
                Projection::Field(field) => Projection::Field(field),
                Projection::Index(index) => Projection::Field(
                    u32::try_from(self.range(&Place::local(index))?.singleton()?).ok()?,
                ),
                Projection::Deref => return None,
            });
        }
        Some(result)
    }

    pub fn range(&self, place: &Place) -> Option<Range> {
        let full = Range::full(self.scalar(place)?);
        if place.is_indirect() {
            return Some(full);
        }
        if let Some(resolved) = self.resolve(place) {
            let own = self.facts.get(&resolved).cloned().unwrap_or(full);
            // An alias holds the same value, so a refinement of its target also bounds it.
            if let Some(Atom::Place(target)) = self.aliases.get(&resolved)
                && let Some(target) = self.facts.get(target)
            {
                let mut both = own.clone();
                if both.restrict(target.lo, target.hi) {
                    return Some(both);
                }
            }
            return Some(own);
        }
        let position = place
            .projection
            .iter()
            .position(|p| matches!(p, Projection::Index(_)))?;
        let Projection::Index(index) = place.projection[position] else {
            unreachable!()
        };
        let range = self.range(&Place::local(index))?;
        if range.lo < 0 || range.hi > 63 {
            return Some(full);
        }
        let mut combined: Option<Range> = None;
        for n in range.lo..=range.hi {
            if !range.contains(n) {
                continue;
            }
            let mut candidate = place.clone();
            candidate.projection[position] = Projection::Field(n as u32);
            let Some(value) = self.range(&candidate) else {
                return Some(full);
            };
            combined = Some(combined.map_or(value.clone(), |previous| previous.join(&value)));
        }
        Some(combined.unwrap_or(full))
    }

    pub fn operand_range(&self, operand: &Operand) -> Option<Range> {
        match operand {
            Operand::Constant(c) => integer(*c).map(Range::exact),
            Operand::Place(p) => self.range(p),
            _ => None,
        }
    }

    pub fn constant(&self, operand: &Operand) -> Option<i128> {
        self.operand_range(operand)?.singleton()
    }

    pub fn symbol(&self, operand: &Operand) -> Option<Atom> {
        if let Some(value) = self.constant(operand) {
            return Some(Atom::Integer(value));
        }
        let Operand::Place(place) = operand else {
            return None;
        };
        self.scalar(place)?;
        let place = self.resolve(place)?;
        Some(
            self.aliases
                .get(&place)
                .cloned()
                .unwrap_or(Atom::Place(place)),
        )
    }

    pub fn atom(&self, atom: &Atom) -> Range {
        match atom {
            Atom::Integer(n) => Range::exact(*n),
            Atom::Place(p) => self.range(p).unwrap(),
            Atom::Length(slice) => self
                .extents
                .get(slice)
                .map_or(Range::full(Scalar::Usize), |&n| Range::exact(i128::from(n))),
        }
    }

    pub fn predicate(&self, operand: &Operand) -> Option<Predicate> {
        let Operand::Place(p) = operand else {
            return None;
        };
        self.predicates.get(&self.resolve(p)?).cloned()
    }

    pub fn operand_value(&self, operand: &Operand) -> Value {
        if let Some(range) = self.operand_range(operand) {
            return Value::scalar(range);
        }
        match operand {
            Operand::AggregateConstant { fields, .. } => {
                let mut result = Value::default();
                for (index, value) in fields.iter().enumerate() {
                    result.field(index as u32, self.operand_value(value));
                }
                result
            }
            Operand::Place(p) => {
                let Some(p) = self.resolve(p) else {
                    return Value::default();
                };
                Value(
                    self.facts
                        .iter()
                        .filter(|(field, _)| p.contains(field))
                        .map(|(field, range)| {
                            (
                                field.projection[p.projection.len()..]
                                    .iter()
                                    .map(|p| {
                                        let Projection::Field(index) = p else {
                                            unreachable!()
                                        };
                                        *index
                                    })
                                    .collect(),
                                range.clone(),
                            )
                        })
                        .collect(),
                )
            }
            _ => Value::default(),
        }
    }

    /// Drops every fact about locals a write through a pointer may have changed.
    pub fn forget_escaped(&mut self) {
        for &local in self.escaped {
            self.forget(&Place::local(local));
        }
    }

    pub fn forget(&mut self, place: &Place) {
        if place.is_indirect() {
            self.forget_escaped();
            return;
        }
        if let Some(destination) = self.resolve(place) {
            self.inherit(&destination);
        }
        let destination = self.resolve(place).unwrap_or_else(|| {
            let mut p = place.clone();
            p.projection.truncate(
                p.projection
                    .iter()
                    .position(|v| matches!(v, Projection::Index(_)))
                    .unwrap(),
            );
            p
        });
        let independent =
            |place: &Place| !destination.contains(place) && !place.contains(&destination);
        let atom = |atom: &Atom| !atom.uses(&destination);
        self.facts.retain(|p, _| independent(p));
        self.aliases
            .retain(|p, a| !destination.contains(p) && atom(a));
        self.predicates
            .retain(|p, v| !destination.contains(p) && !v.uses(&destination));
        self.truths.retain(|p, _| !p.uses(&destination));
        self.slices
            .retain(|view, slice| independent(view) && independent(slice));
        self.copies
            .retain(|copy, source| independent(copy) && independent(source));
        self.extents.retain(|slice, _| independent(slice));
        self.conditions
            .retain(|condition, (_, a, b)| independent(condition) && atom(a) && atom(b));
        self.relations.retain(|(a, _, b)| atom(a) && atom(b));
    }

    pub fn assign(&mut self, destination: &Place, value: Value) {
        let resolved = self.resolve(destination);
        self.forget(destination);
        if let Some(destination) = resolved {
            for (path, range) in value.0 {
                let mut field = destination.clone();
                field
                    .projection
                    .extend(path.into_iter().map(Projection::Field));
                self.facts.insert(field, range);
            }
        }
    }

    /// Joins `other` into this state, widening with `widen`; returns whether the state changed.
    pub fn merge(&mut self, other: &Self, widen: Option<&BTreeSet<i128>>) -> bool {
        let mut own_aliases = self.aliases.iter().peekable();
        let mut other_aliases = other.aliases.iter().peekable();
        let mut updated = Vec::with_capacity(self.facts.len());
        let mut added = Vec::new();
        for (p, own_fact, other_fact) in union(&self.facts, &other.facts) {
            let own = self.fact(p, own_fact, &mut own_aliases);
            let joined = own.join(&other.fact(p, other_fact, &mut other_aliases));
            let range = match widen {
                Some(thresholds) => own.widen(&joined, self.scalar(p).unwrap(), thresholds),
                None => joined,
            };
            if own_fact.is_some() {
                updated.push(range);
            } else {
                added.push((p.clone(), range));
            }
        }
        let mut changed = !added.is_empty();
        for (old, new) in self.facts.values_mut().zip(updated) {
            if *old != new {
                *old = new;
                changed = true;
            }
        }
        self.facts.extend(added);
        changed |= retain_common(&mut self.aliases, &other.aliases);
        changed |= retain_common(&mut self.predicates, &other.predicates);
        changed |= retain_common(&mut self.slices, &other.slices);
        changed |= retain_common(&mut self.copies, &other.copies);
        changed |= retain_common(&mut self.extents, &other.extents);
        changed |= retain_common(&mut self.conditions, &other.conditions);
        let relations = self.relations.len();
        self.relations
            .retain(|relation| other.relations.contains(relation));
        changed |= relations != self.relations.len();
        changed | retain_common(&mut self.truths, &other.truths)
    }

    /// `range(place)` for a fact key; callers visit keys in ascending order with one `aliases`.
    fn fact(&self, place: &Place, own: Option<&Range>, aliases: &mut Aliases<'_>) -> Range {
        while aliases.next_if(|(key, _)| *key < place).is_some() {}
        if place
            .projection
            .iter()
            .any(|p| matches!(p, Projection::Index(_)))
        {
            return self.range(place).unwrap();
        }
        let own = own
            .cloned()
            .unwrap_or_else(|| Range::full(self.scalar(place).unwrap()));
        let alias = aliases.peek().filter(|(key, _)| *key == place);
        if let Some((_, Atom::Place(target))) = alias
            && let Some(target) = self.facts.get(target)
        {
            let mut both = own.clone();
            if both.restrict(target.lo, target.hi) {
                return both;
            }
        }
        own
    }
}

type Aliases<'m> = Peekable<btree_map::Iter<'m, Place, Atom>>;

/// Visits each key of either map once, in ascending order, with its value in each map.
fn union<'m, K: Ord, V>(
    left: &'m BTreeMap<K, V>,
    right: &'m BTreeMap<K, V>,
) -> impl Iterator<Item = (&'m K, Option<&'m V>, Option<&'m V>)> {
    let mut left = left.iter().peekable();
    let mut right = right.iter().peekable();
    std::iter::from_fn(move || {
        let order = match (left.peek(), right.peek()) {
            (Some((a, _)), Some((b, _))) => a.cmp(b),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => return None,
        };
        Some(match order {
            Ordering::Less => left.next().map(|(k, v)| (k, Some(v), None))?,
            Ordering::Greater => right.next().map(|(k, v)| (k, None, Some(v)))?,
            Ordering::Equal => {
                let (k, v) = left.next()?;
                (k, Some(v), right.next().map(|(_, w)| w))
            }
        })
    })
}

/// Keeps the entries `other` holds with an equal value; returns whether any was removed.
fn retain_common<K: Ord, V: PartialEq>(map: &mut BTreeMap<K, V>, other: &BTreeMap<K, V>) -> bool {
    let before = map.len();
    let mut cursor = other.iter().peekable();
    map.retain(|key, value| {
        while cursor.next_if(|(k, _)| *k < key).is_some() {}
        cursor
            .peek()
            .is_some_and(|(k, v)| *k == key && **v == *value)
    });
    map.len() != before
}
