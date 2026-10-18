use super::{
    ast::{Expr, Projection},
    types::{Record, Type},
};

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub struct Local {
    pub name: String,
    pub ty: Type,
    pub mutable: bool,
}

/// An assignable place: a local, the projections from it and the place's type.
pub type Target = (String, Vec<Projection>, Type);

/// The locals, loop labels and const parameter that code at one point can name.
#[derive(Clone, Default)]
pub struct Scope {
    pub locals: Vec<Local>,
    pub labels: Vec<String>,
    pub generic: Option<String>,
    /// Integer parameters, whose values differ between threads.
    pub varying: Vec<String>,
    /// The enclosing helper's `Option` or `Result` result, into which `?` propagates.
    pub question: Option<Type>,
    /// Locals that are not `Copy` though their types pass for it, which closures do not capture:
    /// the `impl Fn` parameters and the `&mut` references that a loop over them freezes.
    pub owned: Vec<String>,
}

impl Scope {
    /// This scope with the given immutable bindings, as a match arm sees it.
    pub fn bind(&self, bindings: &[(String, Type)]) -> Self {
        let mut scope = self.clone();
        scope.locals.extend(bindings.iter().map(|(name, ty)| Local {
            name: name.clone(),
            ty: ty.clone(),
            mutable: false,
        }));
        scope
    }

    /// The locals of enum, `Option` and `Result` types.
    pub fn variants(&self) -> Vec<(String, Type)> {
        self.locals
            .iter()
            .filter(|local| matches!(local.ty, Type::Enum(_) | Type::Option(_) | Type::Result(..)))
            .map(|local| (local.name.clone(), local.ty.clone()))
            .collect()
    }

    /// The closures that code can call: closure locals and `impl Fn` parameters.
    pub fn closures(&self) -> Vec<(String, Vec<Type>, Type)> {
        self.locals
            .iter()
            .filter_map(|local| match &local.ty {
                Type::Closure(parameters, result) => {
                    Some((local.name.clone(), parameters.clone(), (**result).clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// The scope of a closure body with `parameters`: it captures copies of the locals, except
    /// `&mut` references, which a `move` closure would take from the code after it.
    pub fn closure(&self, parameters: &[(String, Type)]) -> Self {
        let mut scope = Self {
            question: None,
            labels: Vec::new(),
            ..self.clone()
        };
        scope.locals.retain(|local| {
            !matches!(local.ty, Type::Ref(_, true) | Type::Slice(_, true))
                && !self.owned.contains(&local.name)
        });
        scope.bind(parameters)
    }

    /// The slice parameters.
    pub fn slices(&self) -> Vec<String> {
        self.locals
            .iter()
            .filter(|local| matches!(local.ty, Type::Slice(..)))
            .map(|local| local.name.clone())
            .collect()
    }

    /// Every place with its type: locals, their projections and what references point to.
    /// Index operands are placeholders.
    pub fn every_place(&self, records: &[Record]) -> Vec<(Expr, Type)> {
        let mut found = Vec::new();
        for local in &self.locals {
            visit(
                records,
                Expr::Local(local.name.clone()),
                &local.ty,
                &mut found,
            );
        }
        found
    }

    /// The places of `ty`.
    pub fn places(&self, records: &[Record], ty: &Type) -> Vec<Expr> {
        self.every_place(records)
            .into_iter()
            .filter(|(_, place)| place == ty)
            .map(|(place, _)| place)
            .collect()
    }

    /// The places of arrays of `element`, with their lengths.
    pub fn arrays(&self, records: &[Record], element: &Type) -> Vec<(Expr, u32)> {
        self.every_place(records)
            .into_iter()
            .filter_map(|(place, ty)| match ty {
                Type::Array(found, length) if *found == *element => Some((place, length)),
                _ => None,
            })
            .collect()
    }

    /// This scope with the local `name` and what it borrows read-only, as the body of a loop
    /// that iterates over it sees it.
    pub fn freeze(&self, name: &str) -> Self {
        let mut scope = self.clone();
        for local in scope.locals.iter_mut().filter(|local| local.name == name) {
            local.mutable = false;
            if let Type::Ref(_, mutable) | Type::Slice(_, mutable) = &mut local.ty {
                if *mutable {
                    scope.owned.push(name.to_owned());
                }
                *mutable = false;
            }
        }
        scope
    }

    /// This scope without the local `name`.
    pub fn hide(&self, name: &str) -> Self {
        let mut scope = self.clone();
        scope.locals.retain(|local| local.name != name);
        scope
    }

    /// The places that an assignment or a `&mut` borrow can target: mutable locals and what
    /// `&mut` parameters point to.
    pub fn targets(&self, records: &[Record]) -> Vec<Target> {
        let mut found = Vec::new();
        for local in &self.locals {
            let (path, ty) = match &local.ty {
                Type::Ref(target, true) => (vec![Projection::Deref], &**target),
                Type::Slice(element, true) => (vec![Projection::Element], &**element),
                ty if local.mutable => (Vec::new(), ty),
                _ => continue,
            };
            projections(records, &local.name, path, ty, &mut found);
        }
        found
    }
}

fn visit(records: &[Record], place: Expr, ty: &Type, found: &mut Vec<(Expr, Type)>) {
    found.push((place.clone(), ty.clone()));
    let zero = || Box::new(Expr::Literal("0".into()));
    match ty {
        Type::Array(element, length) => {
            let index = Expr::Index(Box::new(place), zero(), Some(*length));
            visit(records, index, element, found);
        }
        Type::Tuple(fields) => {
            for (index, field) in fields.iter().enumerate() {
                let projected = Expr::Field(Box::new(place.clone()), index.to_string());
                visit(records, projected, field, found);
            }
        }
        Type::Record(index) => {
            for (name, field) in &records[*index].fields {
                let projected = Expr::Field(Box::new(place.clone()), name.clone());
                visit(records, projected, field, found);
            }
        }
        Type::Ref(target, _) => visit(records, Expr::Deref(Box::new(place)), target, found),
        Type::Slice(element, _) => {
            let index = Expr::Element(Box::new(place), zero(), true);
            visit(records, index, element, found);
        }
        _ => {}
    }
}

fn projections(
    records: &[Record],
    name: &str,
    path: Vec<Projection>,
    ty: &Type,
    found: &mut Vec<Target>,
) {
    found.push((name.to_owned(), path.clone(), ty.clone()));
    let mut child = |projection: Projection, ty: &Type| {
        let mut path = path.clone();
        path.push(projection);
        projections(records, name, path, ty, found);
    };
    match ty {
        Type::Array(element, length) => child(Projection::Index(*length), element),
        Type::Tuple(fields) => {
            for (index, field) in fields.iter().enumerate() {
                child(Projection::Field(index.to_string()), field);
            }
        }
        Type::Record(index) => {
            for (field, ty) in &records[*index].fields {
                child(Projection::Field(field.clone()), ty);
            }
        }
        _ => {}
    }
}
