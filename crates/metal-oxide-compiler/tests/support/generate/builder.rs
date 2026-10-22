use super::{
    Word,
    ast::{Expr, Function, Stmt},
    random::Random,
    scope::{Local, Scope},
    types::{Enum, INTS, Record, Type, U32, zero_sized},
};

pub struct Builder {
    pub random: Random,
    pub records: Vec<Record>,
    pub enums: Vec<Enum>,
    pub helpers: Vec<Function>,
    /// Emits checked operations on values of the thread that fail for some of them: indices past
    /// the length, plain arithmetic and division, and assertions.
    pub fallible: bool,
    /// Calls no helpers or closures and iterates nothing, for code before cooperative operations.
    pub plain: bool,
    names: usize,
}

impl Builder {
    pub fn new(seed: u64) -> Self {
        Self {
            random: Random::new(seed),
            records: Vec::new(),
            enums: Vec::new(),
            helpers: Vec::new(),
            fallible: false,
            plain: false,
            names: 0,
        }
    }

    pub fn fresh(&mut self, prefix: &str) -> String {
        self.names += 1;
        format!("{prefix}{}", self.names)
    }

    /// A type that nests aggregates at most `depth` levels.
    pub fn ty(&mut self, depth: u32) -> Type {
        match self.random.below(15) {
            0..=4 => Type::Int(*self.random.pick(&INTS)),
            5 | 6 => Type::Float,
            7 => Type::Bool,
            8 if depth > 0 => {
                let length = 1 + self.random.below(4) as u32;
                Type::Array(Box::new(self.ty(depth - 1)), length)
            }
            9 if depth > 0 => {
                let fields = (0..1 + self.random.below(3)).map(|_| self.ty(depth - 1));
                Type::Tuple(fields.collect())
            }
            10 | 11 if depth > 0 && !self.records.is_empty() => {
                Type::Record(self.random.below(self.records.len() as u64) as usize)
            }
            12 if depth > 0 && !self.enums.is_empty() => {
                Type::Enum(self.random.below(self.enums.len() as u64) as usize)
            }
            13 if depth > 0 => Type::Option(Box::new(self.ty(depth - 1))),
            14 if depth > 0 => Type::Result(Box::new(self.ty(depth - 1)), Box::new(self.ty(0))),
            _ => Type::Int(*self.random.pick(&INTS)),
        }
    }

    /// An enum whose payloads may hold the records and enums before it.
    pub fn enumeration(&mut self) {
        let variants = (0..1 + self.random.below(4))
            .map(|_| (0..self.random.below(3)).map(|_| self.ty(1)).collect())
            .collect();
        self.enums.push(Enum { variants });
    }

    /// A record whose fields may hold the records and enums before it.
    pub fn record(&mut self) {
        let fields = (0..1 + self.random.below(3))
            .map(|index| (format!("f{index}"), self.ty(1)))
            .collect();
        let name = format!("R{}", self.records.len());
        self.records.push(Record { name, fields });
    }

    /// `count` statements; `nesting` bounds the depth of nested ifs and loops.
    pub fn block(&mut self, scope: &mut Scope, count: u64, nesting: u32) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        for _ in 0..count {
            let stmt = match self.random.below(15) {
                4..=6 => self.assign(scope),
                7 if nesting > 0 => Some(self.branch(scope, nesting)),
                8 if nesting > 0 => Some(self.repeat(scope, nesting)),
                9 if !scope.labels.is_empty() => Some(self.exit(scope)),
                10 if self.fallible => Some(self.assertion(scope)),
                13 => Some(self.declare_closure(scope)),
                14 if nesting > 0 => Some(self.iterate(scope, nesting)),
                11 | 12 => {
                    let mutation = self.mutation(scope);
                    if !mutation.is_empty() {
                        stmts.extend(mutation);
                        continue;
                    }
                    None
                }
                _ => None,
            };
            stmts.push(stmt.unwrap_or_else(|| self.declare(scope)));
        }
        stmts
    }

    /// A local holding a call's result, of a type that a helper borrows, or of a random type.
    fn declare(&mut self, scope: &mut Scope) -> Stmt {
        let call = match self.random.below(6) {
            0 | 1 => self.call_closure(scope, None, 2),
            2 => self.helper_result(scope),
            _ => None,
        };
        if let Some((ty, call)) = call {
            return self.local(scope, ty, call);
        }
        let parameters = self
            .helpers
            .iter()
            .flat_map(|helper| helper.parameters.iter().map(|(_, ty)| ty.clone()))
            .collect::<Vec<_>>();
        let ty = match self.borrowable(&parameters) {
            Some(ty) if self.random.chance(1, 3) => ty,
            _ => self.ty(2),
        };
        let value = self.expr(scope, &ty, 3);
        self.local(scope, ty, value)
    }

    fn local(&mut self, scope: &mut Scope, ty: Type, value: Expr) -> Stmt {
        let name = self.fresh("l");
        scope.locals.push(Local {
            name: name.clone(),
            ty: ty.clone(),
            mutable: true,
        });
        Stmt::Let(name, ty, value)
    }

    /// A call to a helper that only reads its arguments, and its result type.
    fn helper_result(&mut self, scope: &Scope) -> Option<(Type, Expr)> {
        let callees = self
            .helpers
            .iter()
            .enumerate()
            .filter(|(_, helper)| !helper.mutates())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if callees.is_empty() {
            return None;
        }
        let index = *self.random.pick(&callees);
        let call = self.invoke(scope, index, 2)?;
        Some((self.helpers[index].result.clone(), call))
    }

    /// The type of a local that one of `parameters` can borrow, if one is a reference or a
    /// slice.
    fn borrowable(&mut self, parameters: &[Type]) -> Option<Type> {
        let length = 1 + self.random.below(4) as u32;
        let borrowed = parameters
            .iter()
            .filter_map(|ty| match ty {
                Type::Ref(target, _) => Some((**target).clone()),
                Type::Slice(element, _) => Some(Type::Array(element.clone(), length)),
                _ => None,
            })
            .collect::<Vec<_>>();
        if borrowed.is_empty() {
            return None;
        }
        Some(self.random.pick(&borrowed).clone())
    }

    pub fn assign(&mut self, scope: &Scope) -> Option<Stmt> {
        let targets = scope.targets(&self.records);
        if targets.is_empty() {
            return None;
        }
        let target = self.random.pick(&targets).clone();
        let ty = target.2.clone();
        let place = self.target(scope, target);
        let value = self.expr(scope, &ty, 3);
        Some(Stmt::Assign(place, value))
    }

    fn branch(&mut self, scope: &Scope, nesting: u32) -> Stmt {
        let condition = self.expr(scope, &Type::Bool, 2);
        let count = 1 + self.random.below(3);
        let then = self.block(&mut scope.clone(), count, nesting - 1);
        let otherwise = if self.random.chance(1, 2) {
            let count = 1 + self.random.below(3);
            self.block(&mut scope.clone(), count, nesting - 1)
        } else {
            Vec::new()
        };
        Stmt::If(condition, then, otherwise)
    }

    fn repeat(&mut self, scope: &Scope, nesting: u32) -> Stmt {
        let label = self.fresh("o");
        let counter = self.fresh("c");
        let mut inner = scope.clone();
        inner.locals.push(Local {
            name: counter.clone(),
            ty: U32,
            mutable: false,
        });
        inner.labels.push(label.clone());
        let bound = 1 + self.random.below(4);
        let count = 1 + self.random.below(4);
        let body = self.block(&mut inner, count, nesting - 1);
        Stmt::For(
            label,
            counter,
            Expr::Literal(format!("0..{bound}_u32")),
            body,
        )
    }

    fn exit(&mut self, scope: &Scope) -> Stmt {
        let label = self.random.pick(&scope.labels).clone();
        let kind = *self.random.pick(&["break", "continue"]);
        let condition = self.expr(scope, &Type::Bool, 2);
        Stmt::Exit(condition, kind, label)
    }

    /// An assertion, mostly of a condition on a value of the thread, so that it fails for some
    /// threads only.
    fn assertion(&mut self, scope: &Scope) -> Stmt {
        if self.random.chance(1, 4) {
            return Stmt::Assert(self.expr(scope, &Type::Bool, 2));
        }
        Stmt::Assert(self.varying_condition(scope))
    }

    /// A call statement to a helper that writes through `&mut` arguments, after declarations of
    /// locals for it to borrow when the scope has none.
    fn mutation(&mut self, scope: &mut Scope) -> Vec<Stmt> {
        let callees = self
            .helpers
            .iter()
            .enumerate()
            .filter(|(_, helper)| helper.mutates())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if callees.is_empty() {
            return Vec::new();
        }
        let index = *self.random.pick(&callees);
        let parameters = self.helpers[index]
            .parameters
            .iter()
            .map(|(_, ty)| ty.clone())
            .collect::<Vec<_>>();
        let mut stmts = Vec::new();
        for _ in 0..3 {
            if let Some(call) = self.invoke(scope, index, 2) {
                stmts.push(Stmt::Call(call));
                break;
            }
            let Some(ty) = self.borrowable(&parameters) else {
                break;
            };
            let value = self.expr(scope, &ty, 3);
            stmts.push(self.local(scope, ty, value));
        }
        stmts
    }

    /// A helper parameter: mostly a value, else a reference or a slice, which may be `&mut`, or
    /// an `impl Fn`.
    fn parameter(&mut self) -> Type {
        match self.random.below(9) {
            0 => Type::Ref(Box::new(self.ty(1)), self.random.chance(1, 2)),
            1 => {
                let element = self.ty(1);
                if zero_sized(&element, &self.records, &self.enums) {
                    return element;
                }
                Type::Slice(Box::new(element), self.random.chance(1, 2))
            }
            2 => {
                let parameters = (0..self.random.below(3)).map(|_| self.ty(1)).collect();
                Type::Closure(parameters, Box::new(self.ty(1)))
            }
            _ => self.ty(1),
        }
    }

    /// A helper that the helpers after it and `run` may call; it calls only earlier ones.
    pub fn helper(&mut self) {
        let name = format!("h{}", self.helpers.len());
        let generic = self.random.chance(1, 4).then(|| "K".to_owned());
        let parameters = (0..1 + self.random.below(3))
            .map(|index| (format!("p{index}"), self.parameter()))
            .collect::<Vec<_>>();
        let result = self.ty(1);
        let mut scope = Scope {
            generic: generic.clone(),
            question: matches!(result, Type::Option(_) | Type::Result(..)).then(|| result.clone()),
            varying: parameters
                .iter()
                .filter(|(_, ty)| matches!(ty, Type::Int(_)))
                .map(|(name, _)| name.clone())
                .collect(),
            owned: parameters
                .iter()
                .filter(|(_, ty)| matches!(ty, Type::Closure(..)))
                .map(|(name, _)| name.clone())
                .collect(),
            ..Scope::default()
        };
        for (name, ty) in &parameters {
            scope.locals.push(Local {
                name: name.clone(),
                ty: ty.clone(),
                mutable: false,
            });
        }
        let count = self.random.below(4);
        let body = self.block(&mut scope, count, 1);
        let value = self.expr(&scope, &result, 3);
        self.helpers.push(Function {
            name,
            generic,
            parameters,
            result,
            body,
            value,
        });
    }

    /// `run(input, k0, ..) -> [u32; words]`, the code that every thread executes, which also
    /// takes the `values` that `cooperate` computed.
    pub fn run(&mut self, words: u32, values: u32) -> (Function, Vec<Word>) {
        let parameters = std::iter::once("input".to_owned())
            .chain((0..values).map(|value| format!("k{value}")))
            .map(|name| (name, U32))
            .collect::<Vec<_>>();
        let mut scope = Scope {
            varying: parameters.iter().map(|(name, _)| name.clone()).collect(),
            ..Scope::default()
        };
        for (name, ty) in &parameters {
            scope.locals.push(Local {
                name: name.clone(),
                ty: ty.clone(),
                mutable: false,
            });
        }
        let count = 3 + self.random.below(5);
        let mut body = self.block(&mut scope, count, 2);
        let mut values = Vec::new();
        let mut kinds = Vec::new();
        for _ in 0..words {
            let ty = self.ty(0);
            let mut value = Box::new(self.expr(&scope, &ty, 3));
            if matches!(ty, Type::Int(int) if int.bits == 64) {
                let name = self.fresh("word");
                body.push(Stmt::Let(name.clone(), ty.clone(), *value));
                value = Box::new(Expr::Local(name));
                values.push(Expr::Cast(value.clone(), U32));
                kinds.push(Word::Exact);
                value = Box::new(Expr::Binary(
                    value,
                    ">>",
                    Box::new(Expr::Literal("32_u32".into())),
                ));
            }
            let (word, kind) = match ty {
                Type::Float => (Expr::Method(value, "to_bits", vec![]), Word::Float),
                _ => (Expr::Cast(value, U32), Word::Exact),
            };
            values.push(word);
            kinds.push(kind);
        }
        let result = Type::Array(Box::new(U32), values.len() as u32);
        let run = Function {
            name: "run".into(),
            generic: None,
            parameters,
            result: result.clone(),
            body,
            value: Expr::Aggregate(result, values),
        };
        (run, kinds)
    }
}
