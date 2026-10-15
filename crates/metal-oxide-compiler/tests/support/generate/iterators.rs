use super::{
    ast::{Expr, Stmt},
    builder::Builder,
    scope::{Local, Scope},
    types::{Type, U32, zero_sized},
};

/// An iterator, the type of its items and the local whose array it borrows.
type Iterator = (Expr, Type, Option<String>);

impl Builder {
    /// A value of `ty` that an iterator computes: a fold, `any`, `all`, a count, `max` or `min`.
    pub fn iterated(&mut self, scope: &Scope, ty: &Type, depth: u32) -> Expr {
        let (iterator, item, _) = self.iterator(scope, depth);
        let iterator = Box::new(iterator);
        match ty {
            Type::Bool if self.random.chance(1, 2) => {
                let method = *self.random.pick(&["any", "all"]);
                let predicate = self.closure(scope, &[item], &Type::Bool, depth);
                Expr::Method(iterator, method, vec![predicate])
            }
            Type::Int(_) if self.random.chance(1, 3) => {
                let count = Expr::Method(iterator, "count", Vec::new());
                Expr::Cast(Box::new(count), ty.clone())
            }
            Type::Int(_) if item == *ty && self.random.chance(1, 2) => {
                let method = *self.random.pick(&["max", "min"]);
                let default = self.expr(scope, ty, depth);
                let extreme = Expr::Method(iterator, method, Vec::new());
                Expr::Method(Box::new(extreme), "unwrap_or", vec![default])
            }
            _ => {
                let initial = self.expr(scope, ty, depth);
                let step = self.closure(scope, &[ty.clone(), item], ty, depth);
                Expr::Method(iterator, "fold", vec![initial, step])
            }
        }
    }

    /// `for item in iterator { .. }`, whose body reads but does not write the iterated local,
    /// or `for item in array.iter_mut() { .. }`, whose body writes through `item` alone.
    pub fn iterate(&mut self, scope: &Scope, nesting: u32) -> Stmt {
        let label = self.fresh("o");
        let binding = self.fresh("x");
        let arrays = scope
            .targets(&self.records)
            .into_iter()
            .filter(|(_, _, ty)| self.sliceable(ty))
            .collect::<Vec<_>>();
        let (iterable, item, mut inner) = if !arrays.is_empty() && self.random.chance(1, 3) {
            let target = self.random.pick(&arrays).clone();
            let (root, Type::Array(element, _)) = (target.0.clone(), target.2.clone()) else {
                unreachable!()
            };
            let array = self.target(scope, target);
            let iterable = self.adapted(Expr::Method(Box::new(array), "iter_mut", Vec::new()));
            (iterable, Type::Ref(element, true), scope.hide(&root))
        } else {
            let (iterable, item, root) = self.iterator(scope, 2);
            let inner = match root {
                Some(root) => scope.freeze(&root),
                None => scope.clone(),
            };
            (iterable, item, inner)
        };
        inner.labels.push(label.clone());
        inner.locals.push(Local {
            name: binding.clone(),
            ty: item,
            mutable: false,
        });
        let count = 1 + self.random.below(3);
        let body = self.block(&mut inner, count, nesting - 1);
        Stmt::For(label, binding, iterable, body)
    }

    /// An iterator over an array place or a short range, with adapters.
    fn iterator(&mut self, scope: &Scope, depth: u32) -> Iterator {
        let arrays = scope
            .every_place(&self.records)
            .into_iter()
            .filter(|(_, ty)| self.sliceable(ty))
            .collect::<Vec<_>>();
        let (mut iterator, mut item, root) = if !arrays.is_empty() && self.random.chance(3, 4) {
            let (array, Type::Array(element, _)) = self.random.pick(&arrays).clone() else {
                unreachable!()
            };
            let array = self.indexed(scope, array);
            let root = root(&array);
            let iterator = Expr::Method(Box::new(array), "iter", Vec::new());
            if self.random.chance(1, 3) {
                let copied = Expr::Method(Box::new(iterator), "copied", Vec::new());
                (copied, *element, root)
            } else {
                (iterator, Type::Ref(element, false), root)
            }
        } else {
            let end = 1 + self.random.below(8);
            (Expr::Literal(format!("(0_u32..{end})")), U32, None)
        };
        iterator = self.adapted(iterator);
        for _ in 0..self.random.below(2) {
            iterator = if self.random.chance(1, 2) {
                let mapped = self.ty(1);
                let map = self.closure(scope, &[item], &mapped, depth);
                item = mapped;
                Expr::Method(Box::new(iterator), "map", vec![map])
            } else {
                let reference = Type::Ref(Box::new(item.clone()), false);
                let predicate = self.closure(scope, &[reference], &Type::Bool, depth);
                Expr::Method(Box::new(iterator), "filter", vec![predicate])
            };
        }
        (iterator, item, root)
    }

    /// Whether `ty` is an array whose elements a slice can hold.
    fn sliceable(&self, ty: &Type) -> bool {
        matches!(ty, Type::Array(element, _) if !zero_sized(element, &self.records, &self.enums))
    }

    /// `iterator` reversed, then skipping, taking or stepping; the items keep their type.
    fn adapted(&mut self, iterator: Expr) -> Expr {
        let mut iterator = iterator;
        if self.random.chance(1, 3) {
            iterator = Expr::Method(Box::new(iterator), "rev", Vec::new());
        }
        for _ in 0..self.random.below(3) {
            let (method, count) = match self.random.below(3) {
                0 => ("skip", self.random.below(3)),
                1 => ("take", 1 + self.random.below(4)),
                _ => ("step_by", 1 + self.random.below(3)),
            };
            let count = Expr::Literal(count.to_string());
            iterator = Expr::Method(Box::new(iterator), method, vec![count]);
        }
        iterator
    }
}

/// The local that a place expression starts from.
pub(super) fn root(place: &Expr) -> Option<String> {
    match place {
        Expr::Local(name) => Some(name.clone()),
        Expr::Field(inner, _)
        | Expr::Index(inner, ..)
        | Expr::Element(inner, ..)
        | Expr::Deref(inner) => root(inner),
        _ => None,
    }
}
