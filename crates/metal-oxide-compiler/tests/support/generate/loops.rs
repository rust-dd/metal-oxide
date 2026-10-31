use super::{
    ast::{Expr, Stmt},
    builder::Builder,
    scope::{Local, Scope},
    types::{Int, Type, U32},
};

impl Builder {
    /// `for counter in range { .. }`, over a literal range or one whose bounds differ between
    /// threads.
    pub fn repeat(&mut self, scope: &Scope, nesting: u32) -> Stmt {
        let label = self.fresh("o");
        let counter = self.fresh("c");
        let range = if self.random.chance(1, 2) {
            Expr::Literal(format!("0..{}_u32", 1 + self.random.below(4)))
        } else {
            self.range(scope, 4)
        };
        let mut inner = scope.clone();
        inner.locals.push(Local {
            name: counter.clone(),
            ty: U32,
            mutable: false,
        });
        inner.labels.push(label.clone());
        let count = 1 + self.random.below(4);
        let body = self.block(&mut inner, count, nesting - 1);
        Stmt::For(label, counter, range, body)
    }

    /// `while counter < bound && condition { counter += 1; .. }`; later code reads the counter.
    pub fn repeat_while(&mut self, scope: &mut Scope, nesting: u32) -> Stmt {
        let label = self.fresh("o");
        let counter = self.fresh("w");
        let bound = self.bound(scope, 5);
        let condition = self.expr(scope, &Type::Bool, 2);
        let local = Local {
            name: counter.clone(),
            ty: U32,
            mutable: false,
        };
        let mut inner = scope.clone();
        inner.locals.push(local.clone());
        inner.labels.push(label.clone());
        let count = 1 + self.random.below(3);
        let body = self.block(&mut inner, count, nesting - 1);
        scope.locals.push(local);
        Stmt::While(label, counter, bound, condition, body)
    }

    pub fn exit(&mut self, scope: &Scope) -> Stmt {
        let label = self.random.pick(&scope.labels).clone();
        let kind = *self.random.pick(&["break", "continue"]);
        let condition = self.expr(scope, &Type::Bool, 2);
        Stmt::Exit(condition, kind, label)
    }

    /// `if condition { ..; return value; }`, mostly on a value of the thread, so that some
    /// threads leave the function early.
    pub fn early_return(&mut self, scope: &Scope, nesting: u32) -> Option<Stmt> {
        let result = scope.returns.clone()?;
        let condition = if self.random.chance(1, 4) {
            self.expr(scope, &Type::Bool, 2)
        } else {
            self.varying_condition(scope)
        };
        let mut inner = scope.clone();
        let count = self.random.below(3);
        let mut then = self.block(&mut inner, count, nesting.saturating_sub(1));
        then.push(Stmt::Return(self.expr(&inner, &result, 3)));
        Some(Stmt::If(condition, then, Vec::new()))
    }

    /// A `u32` range of at most `limit` values: `start..end` or `start..=end`, whose bounds may
    /// differ between threads, so that a range can be empty or reversed.
    pub fn range(&mut self, scope: &Scope, limit: u32) -> Expr {
        let zero = || Box::new(Expr::Literal("0_u32".into()));
        let (start, operator, end) = match self.random.below(4) {
            0 => (
                Box::new(self.bound(scope, 3)),
                "..",
                self.bound(scope, limit + 1),
            ),
            1 => (zero(), "..=", self.bound(scope, limit)),
            _ => (zero(), "..", self.bound(scope, limit + 1)),
        };
        Expr::Binary(start, operator, Box::new(end))
    }

    /// A `u32` below `limit`: a literal, or a value of the thread taken modulo `limit`.
    fn bound(&mut self, scope: &Scope, limit: u32) -> Expr {
        if self.random.chance(1, 3) {
            let value = self.random.below(u64::from(limit));
            return Expr::Literal(format!("{value}_u32"));
        }
        let value = self.expr(scope, &U32, 2);
        let varying = self.varying(scope, Int::U32);
        let mixed = Expr::Binary(Box::new(value), "^", Box::new(varying));
        let limit = Expr::Literal(format!("{limit}_u32"));
        Expr::Binary(Box::new(mixed), "%", Box::new(limit))
    }
}
