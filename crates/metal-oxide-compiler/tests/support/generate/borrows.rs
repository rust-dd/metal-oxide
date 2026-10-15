use super::{
    ast::{Bound, Expr, Projection},
    builder::Builder,
    scope::{Scope, Target},
    types::{INTS, Type},
};

impl Builder {
    /// Arguments for `parameters`, or `None` when the scope has no place to borrow for one of
    /// them. Every local that a `&mut` argument borrows is named by no other argument.
    pub fn arguments(
        &mut self,
        scope: &Scope,
        parameters: &[(String, Type)],
        depth: u32,
    ) -> Option<Vec<Expr>> {
        let mut values = Vec::new();
        for (_, ty) in parameters {
            values.push(match ty {
                Type::Ref(_, true) | Type::Slice(_, true) => None,
                Type::Ref(target, false) => Some(self.shared(scope, target)?),
                Type::Slice(element, false) => Some(self.shared_slice(scope, element)?),
                Type::Closure(parameters, result) => {
                    Some(self.closure(scope, parameters, result, depth))
                }
                ty => Some(self.expr(scope, ty, depth)),
            });
        }
        let mut borrowed = Vec::new();
        for (index, (_, ty)) in parameters.iter().enumerate() {
            let (wanted, slice) = match ty {
                Type::Ref(target, true) => (&**target, false),
                Type::Slice(element, true) => (&**element, true),
                _ => continue,
            };
            let candidates = scope
                .targets(&self.records)
                .into_iter()
                .filter(|(name, _, ty)| {
                    let fits = match ty {
                        Type::Array(element, _) if slice => **element == *wanted,
                        ty => !slice && ty == wanted,
                    };
                    fits && !values.iter().flatten().any(|value| value.mentions(name))
                })
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                return None;
            }
            let target = self.random.pick(&candidates).clone();
            let root = target.0.clone();
            let length = match &target.2 {
                Type::Array(_, length) => *length,
                _ => 0,
            };
            let place = self.target(scope, target);
            values[index] = Some(if slice {
                self.subslice(scope, place, length, true)
            } else {
                Expr::Borrow(Box::new(place), true)
            });
            borrowed.push((index, root));
        }
        let values = values.into_iter().map(Option::unwrap).collect::<Vec<_>>();
        // Index operands of a borrowed place may name what another argument borrows.
        let conflict = borrowed.iter().any(|(index, root)| {
            values
                .iter()
                .enumerate()
                .any(|(other, value)| other != *index && value.mentions(root))
        });
        (!conflict).then_some(values)
    }

    /// `&place` for a place of `target`.
    fn shared(&mut self, scope: &Scope, target: &Type) -> Option<Expr> {
        let places = scope.places(&self.records, target);
        if places.is_empty() {
            return None;
        }
        let place = self.random.pick(&places).clone();
        Some(Expr::Borrow(Box::new(self.indexed(scope, place)), false))
    }

    /// A shared subslice of an array of `element`.
    fn shared_slice(&mut self, scope: &Scope, element: &Type) -> Option<Expr> {
        let arrays = scope.arrays(&self.records, element);
        if arrays.is_empty() {
            return None;
        }
        let (array, length) = self.random.pick(&arrays).clone();
        let array = self.indexed(scope, array);
        Some(self.subslice(scope, array, length, false))
    }

    /// The place of an assignment or `&mut` target, with index operands; an index that names
    /// the target's local itself is a literal.
    pub fn target(&mut self, scope: &Scope, (name, path, _): Target) -> Expr {
        let mut place = Expr::Local(name.clone());
        for projection in path {
            let index = |this: &mut Self| {
                let int = *this.random.pick(&INTS);
                let index = this.expr(scope, &Type::Int(int), 1);
                let index = if index.mentions(&name) {
                    Expr::Literal("0".into())
                } else {
                    index
                };
                Box::new(index)
            };
            place = match projection {
                Projection::Field(field) => Expr::Field(Box::new(place), field),
                Projection::Index(length) => {
                    Expr::Index(Box::new(place), index(self), Some(length))
                }
                Projection::Deref => Expr::Deref(Box::new(place)),
                Projection::Element => Expr::Element(Box::new(place), index(self), true),
            };
        }
        place
    }

    /// `&array[start..end]`: nonempty and in bounds, or in fallible programs with bounds from
    /// values of the thread, which can fail the slice checks or leave the slice empty.
    fn subslice(&mut self, scope: &Scope, array: Expr, length: u32, mutable: bool) -> Expr {
        // A bound that reads the array would use it while `&mut` borrows it.
        let borrowed = mutable.then(|| super::iterators::root(&array)).flatten();
        let array = Box::new(array);
        if self.fallible && self.random.chance(1, 3) {
            let bound = |this: &mut Self| {
                let int = *this.random.pick(&INTS);
                let index = this.varying(scope, int);
                if borrowed.as_ref().is_some_and(|name| index.mentions(name)) {
                    Bound::Constant(0)
                } else {
                    Bound::Varying(Box::new(index), length + 2)
                }
            };
            let start = if self.random.chance(1, 2) {
                bound(self)
            } else {
                Bound::Constant(0)
            };
            return Expr::Subslice(array, start, bound(self), mutable);
        }
        let start = self.random.below(u64::from(length)) as u32;
        let end = start + 1 + self.random.below(u64::from(length - start)) as u32;
        Expr::Subslice(array, Bound::Constant(start), Bound::Constant(end), mutable)
    }
}
