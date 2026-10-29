use super::{
    ast::{Expr, Stmt},
    builder::Builder,
    scope::{Local, Scope},
    types::Type,
};

impl Builder {
    /// `move |a1: T1, ..| -> result { body }`, whose body captures copies of the scope's locals.
    pub fn closure(
        &mut self,
        scope: &Scope,
        parameters: &[Type],
        result: &Type,
        depth: u32,
    ) -> Expr {
        let parameters = parameters
            .iter()
            .map(|ty| (self.fresh("a"), ty.clone()))
            .collect::<Vec<_>>();
        let body = self.expr(&scope.closure(&parameters), result, depth);
        Expr::Closure(parameters, result.clone(), Box::new(body))
    }

    /// `let name = move |..| ..;`, a closure that later code can call.
    pub fn declare_closure(&mut self, scope: &mut Scope) -> Stmt {
        let parameters = (0..self.random.below(3))
            .map(|_| self.ty(1))
            .collect::<Vec<_>>();
        let result = self.ty(1);
        let closure = self.closure(scope, &parameters, &result, 3);
        let name = self.fresh("c");
        scope.locals.push(Local {
            name: name.clone(),
            ty: Type::Closure(parameters, Box::new(result)),
            mutable: false,
        });
        Stmt::Closure(name, closure)
    }

    /// A call to a closure in scope that returns `ty`, or any closure, and its result type.
    pub fn call_closure(
        &mut self,
        scope: &Scope,
        ty: Option<&Type>,
        depth: u32,
    ) -> Option<(Type, Expr)> {
        let closures = scope
            .closures()
            .into_iter()
            .filter(|(_, _, result)| ty.is_none_or(|ty| result == ty))
            .collect::<Vec<_>>();
        if closures.is_empty() {
            return None;
        }
        let (name, parameters, result) = self.random.pick(&closures).clone();
        let arguments = parameters
            .iter()
            .map(|ty| self.expr(scope, ty, depth))
            .collect::<Vec<_>>();
        Some((result, Expr::Call(name, None, arguments)))
    }
}
