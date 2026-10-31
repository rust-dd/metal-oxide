use super::{
    ast::Expr,
    builder::Builder,
    scope::Scope,
    types::{Type, render},
};

/// A pattern and the payload bindings it introduces.
type Arm = (String, Vec<(String, Type)>);

impl Builder {
    /// A constructed value of the enum, `Option` or `Result` type `ty`.
    pub fn variant(&mut self, scope: &Scope, ty: &Type, depth: u32) -> Expr {
        match ty {
            Type::Option(value) => match (self.random.below(5), &**value) {
                (0, _) => {
                    let value = render(value, &self.records);
                    Expr::Variant(format!("None::<{value}>"), Vec::new())
                }
                (1, _) => {
                    let condition = self.expr(scope, &Type::Bool, depth);
                    let value = self.expr(scope, value, depth);
                    Expr::Method(Box::new(condition), "then_some", vec![value])
                }
                (2, Type::Int(int)) => self.checked(scope, *int, depth),
                _ => Expr::Variant("Some".into(), vec![self.expr(scope, value, depth)]),
            },
            Type::Result(value, error) => {
                let types = format!(
                    "::<{}, {}>",
                    render(value, &self.records),
                    render(error, &self.records)
                );
                if self.random.chance(1, 3) {
                    Expr::Variant(format!("Err{types}"), vec![self.expr(scope, error, depth)])
                } else {
                    Expr::Variant(format!("Ok{types}"), vec![self.expr(scope, value, depth)])
                }
            }
            Type::Enum(index) => {
                let variants = &self.enums[*index].variants;
                let variant = self.random.below(variants.len() as u64) as usize;
                let payload = variants[variant].clone();
                let payload = payload
                    .iter()
                    .map(|ty| self.expr(scope, ty, depth))
                    .collect::<Vec<_>>();
                Expr::Variant(format!("E{index}::V{variant}"), payload)
            }
            _ => unreachable!("{ty:?} has no variants"),
        }
    }

    /// A value of `ty` taken out of a variant: a `match` that binds payloads, `unwrap_or`, `?`
    /// where the helper propagates, and in fallible programs `unwrap` and `expect` of options
    /// that are `None` for some threads.
    pub fn unpack(&mut self, scope: &Scope, ty: &Type, depth: u32) -> Expr {
        let propagated = match &scope.question {
            Some(Type::Option(_)) => Some(Type::Option(Box::new(ty.clone()))),
            Some(Type::Result(_, error)) => Some(Type::Result(Box::new(ty.clone()), error.clone())),
            _ => None,
        };
        if let Some(propagated) = propagated
            && self.random.chance(1, 4)
        {
            return Expr::Try(Box::new(self.expr(scope, &propagated, depth)));
        }
        match self.random.below(4) {
            0 if self.fallible => {
                let condition = self.varying_condition(scope);
                let value = self.expr(scope, ty, depth);
                let option = Expr::Method(Box::new(condition), "then_some", vec![value]);
                match self.random.below(3) {
                    0 => Expr::Method(Box::new(option), "unwrap", Vec::new()),
                    1 => {
                        let message = Expr::Literal("\"generated expectation\"".into());
                        Expr::Method(Box::new(option), "expect", vec![message])
                    }
                    _ => {
                        let error = self.ty(0);
                        let error = self.expr(scope, &error, depth);
                        let result = Expr::Method(Box::new(option), "ok_or", vec![error]);
                        Expr::Method(Box::new(result), "unwrap", Vec::new())
                    }
                }
            }
            1 => {
                let option = self.expr(scope, &Type::Option(Box::new(ty.clone())), depth);
                let default = self.expr(scope, ty, depth);
                Expr::Method(Box::new(option), "unwrap_or", vec![default])
            }
            _ => self.matched(scope, ty, depth),
        }
    }

    /// Whether a value of a random variant type has some variant.
    pub fn tested(&mut self, scope: &Scope, depth: u32) -> Expr {
        let (value, source) = self.inspected(scope, depth);
        let value = Box::new(value);
        // `is_some` borrows its receiver, and Rust promotes a borrowed constant, which the
        // compiler rejects; places are never constants.
        let place = matches!(*value, Expr::Local(_) | Expr::Field(..) | Expr::Index(..));
        match source {
            Type::Option(_) if place => {
                Expr::Method(value, self.random.pick(&["is_some", "is_none"]), Vec::new())
            }
            Type::Result(..) if place => {
                Expr::Method(value, self.random.pick(&["is_ok", "is_err"]), Vec::new())
            }
            _ => {
                let arms = self.arms(&source, false);
                let (pattern, _) = self.random.pick(&arms).clone();
                Expr::Matches(value, pattern)
            }
        }
    }

    /// A `match` on a value of a random variant type, whose arms bind its payloads.
    fn matched(&mut self, scope: &Scope, ty: &Type, depth: u32) -> Expr {
        let (scrutinee, source) = self.inspected(scope, depth);
        let mut arms = self.arms(&source, true);
        if arms.len() > 1 && self.random.chance(1, 3) {
            *arms.last_mut().unwrap() = ("_".into(), Vec::new());
        }
        let arms = arms
            .into_iter()
            .map(|(pattern, bindings)| (pattern, self.expr(&scope.bind(&bindings), ty, depth)))
            .collect::<Vec<_>>();
        Expr::MatchVariant(Box::new(scrutinee), arms)
    }

    /// A value to inspect: often a local of a variant type, else a value of a random one.
    fn inspected(&mut self, scope: &Scope, depth: u32) -> (Expr, Type) {
        let locals = scope.variants();
        if !locals.is_empty() && self.random.chance(1, 2) {
            let (name, ty) = self.random.pick(&locals).clone();
            return (Expr::Local(name), ty);
        }
        let ty = self.variant_type();
        (self.expr(scope, &ty, depth), ty)
    }

    fn variant_type(&mut self) -> Type {
        match self.random.below(3) {
            0 if !self.enums.is_empty() => {
                Type::Enum(self.random.below(self.enums.len() as u64) as usize)
            }
            1 => Type::Result(Box::new(self.ty(1)), Box::new(self.ty(0))),
            _ => Type::Option(Box::new(self.ty(1))),
        }
    }

    /// A pattern for each variant of `ty`, which binds the payloads when `bind` is set.
    fn arms(&mut self, ty: &Type, bind: bool) -> Vec<Arm> {
        let variants = match ty {
            Type::Option(value) => vec![
                ("Some".into(), vec![(**value).clone()]),
                ("None".into(), Vec::new()),
            ],
            Type::Result(value, error) => vec![
                ("Ok".into(), vec![(**value).clone()]),
                ("Err".into(), vec![(**error).clone()]),
            ],
            Type::Enum(index) => self.enums[*index]
                .variants
                .iter()
                .enumerate()
                .map(|(variant, payload)| (format!("E{index}::V{variant}"), payload.clone()))
                .collect(),
            _ => unreachable!("{ty:?} has no variants"),
        };
        variants
            .into_iter()
            .map(|(name, payload)| match payload.len() {
                0 => (name, Vec::new()),
                _ if !bind => (format!("{name}(..)"), Vec::new()),
                _ => {
                    let bindings = payload
                        .into_iter()
                        .map(|ty| (self.fresh("b"), ty))
                        .collect::<Vec<_>>();
                    let names = bindings
                        .iter()
                        .map(|(name, _)| name.as_str())
                        .collect::<Vec<_>>();
                    (format!("{name}({})", names.join(", ")), bindings)
                }
            })
            .collect()
    }
}
