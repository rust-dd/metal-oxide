use std::collections::BTreeMap;

use super::{
    domain::{Range, integer},
    predicate::{Atom, Predicate},
};
use crate::*;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
    pub facts: BTreeMap<Place, Range>,
    pub aliases: BTreeMap<Place, Atom>,
    pub predicates: BTreeMap<Place, Predicate>,
    pub truths: BTreeMap<Predicate, bool>,
}

impl PartialEq for State<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.facts == other.facts
            && self.aliases == other.aliases
            && self.predicates == other.predicates
            && self.truths == other.truths
    }
}

impl Eq for State<'_> {}

impl<'a> State<'a> {
    pub fn new(module: &'a Module, function: &'a Function) -> Self {
        Self {
            module,
            function,
            facts: BTreeMap::new(),
            aliases: BTreeMap::new(),
            predicates: BTreeMap::new(),
            truths: BTreeMap::new(),
        }
    }

    pub fn scalar(&self, place: &Place) -> Option<Scalar> {
        match place_type(self.module, self.function, place, &self.function.source).ok()? {
            Type::Scalar(s) if s.is_integer() || s == Scalar::Bool => Some(s),
            _ => None,
        }
    }

    pub fn resolve(&self, place: &Place) -> Option<Place> {
        let mut result = Place::local(place.local);
        for projection in &place.projection {
            result.projection.push(match *projection {
                Projection::Field(field) => Projection::Field(field),
                Projection::Index(index) => Projection::Field(
                    u32::try_from(self.range(&Place::local(index))?.singleton()?).ok()?,
                ),
            });
        }
        Some(result)
    }

    pub fn range(&self, place: &Place) -> Option<Range> {
        let full = Range::full(self.scalar(place)?);
        if let Some(resolved) = self.resolve(place) {
            return Some(self.facts.get(&resolved).cloned().unwrap_or(full));
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

    pub fn forget(&mut self, place: &Place) {
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
        self.facts
            .retain(|p, _| !destination.contains(p) && !p.contains(&destination));
        self.aliases.retain(|p, a| !destination.contains(p) && !matches!(a, Atom::Place(p) if destination.contains(p) || p.contains(&destination)));
        self.predicates
            .retain(|p, v| !destination.contains(p) && !v.uses(&destination));
        self.truths.retain(|p, _| !p.uses(&destination));
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

    pub fn merge(&self, other: &Self, widen: bool) -> Self {
        let mut result = Self::new(self.module, self.function);
        for p in self.facts.keys().chain(other.facts.keys()) {
            if result.facts.contains_key(p) {
                continue;
            }
            let a = self.range(p).unwrap();
            let b = other.range(p).unwrap();
            let joined = a.join(&b);
            result.facts.insert(
                p.clone(),
                if widen {
                    a.widen(&joined, self.scalar(p).unwrap())
                } else {
                    joined
                },
            );
        }
        for (p, a) in &self.aliases {
            if other.aliases.get(p) == Some(a) {
                result.aliases.insert(p.clone(), a.clone());
            }
        }
        for (p, v) in &self.predicates {
            if other.predicates.get(p) == Some(v) {
                result.predicates.insert(p.clone(), v.clone());
            }
        }
        for (p, v) in &self.truths {
            if other.truths.get(p) == Some(v) {
                result.truths.insert(p.clone(), *v);
            }
        }
        result
    }
}
