use super::{
    Word,
    ast::{Expr, Function, Projection, Stmt},
    random::Random,
    types::{INTS, Int, Record, Type},
};

const U32: Type = Type::Int(Int {
    bits: 32,
    signed: false,
});

pub struct Builder {
    pub random: Random,
    pub records: Vec<Record>,
    pub helpers: Vec<Function>,
    /// Emits checked operations on values of the thread that fail for some of them: indices past
    /// the length, plain arithmetic and division, and assertions.
    pub fallible: bool,
    names: usize,
}

#[derive(Clone)]
struct Local {
    name: String,
    ty: Type,
    mutable: bool,
}

/// The locals, loop labels and const parameter that code at one point can name.
#[derive(Clone, Default)]
pub struct Scope {
    locals: Vec<Local>,
    labels: Vec<String>,
    pub generic: Option<String>,
    /// Integer parameters, whose values differ between threads.
    pub varying: Vec<String>,
}

impl Scope {
    /// The locals and their projections of `ty`; index operands are placeholders.
    pub fn places(&self, records: &[Record], ty: &Type) -> Vec<Expr> {
        let mut found = Vec::new();
        for local in &self.locals {
            collect(
                records,
                Expr::Local(local.name.clone()),
                &local.ty,
                ty,
                &mut found,
            );
        }
        found
    }

    /// Places of mutable locals that an assignment can target, with their types.
    fn targets(&self, records: &[Record]) -> Vec<(String, Vec<Projection>, Type)> {
        let mut found = Vec::new();
        for local in self.locals.iter().filter(|local| local.mutable) {
            projections(records, &local.name, Vec::new(), &local.ty, &mut found);
        }
        found
    }
}

fn collect(records: &[Record], place: Expr, ty: &Type, wanted: &Type, found: &mut Vec<Expr>) {
    if ty == wanted {
        found.push(place.clone());
    }
    match ty {
        Type::Array(element, length) => {
            let zero = Box::new(Expr::Literal("0".into()));
            let index = Expr::Index(Box::new(place), zero, Some(*length));
            collect(records, index, element, wanted, found);
        }
        Type::Tuple(fields) => {
            for (index, field) in fields.iter().enumerate() {
                let projected = Expr::Field(Box::new(place.clone()), index.to_string());
                collect(records, projected, field, wanted, found);
            }
        }
        Type::Record(index) => {
            for (name, field) in &records[*index].fields {
                let projected = Expr::Field(Box::new(place.clone()), name.clone());
                collect(records, projected, field, wanted, found);
            }
        }
        _ => {}
    }
}

fn projections(
    records: &[Record],
    name: &str,
    path: Vec<Projection>,
    ty: &Type,
    found: &mut Vec<(String, Vec<Projection>, Type)>,
) {
    found.push((name.to_owned(), path.clone(), ty.clone()));
    let mut child = |projection: Projection, ty: &Type| {
        let mut path = path.clone();
        path.push(projection);
        projections(records, name, path, ty, found);
    };
    match ty {
        Type::Array(element, length) => {
            child(
                Projection::Index(Expr::Literal("0".into()), *length),
                element,
            );
        }
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

impl Builder {
    pub fn new(seed: u64) -> Self {
        Self {
            random: Random::new(seed),
            records: Vec::new(),
            helpers: Vec::new(),
            fallible: false,
            names: 0,
        }
    }

    fn fresh(&mut self, prefix: &str) -> String {
        self.names += 1;
        format!("{prefix}{}", self.names)
    }

    /// A type that nests aggregates at most `depth` levels.
    pub fn ty(&mut self, depth: u32) -> Type {
        match self.random.below(12) {
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
            _ => Type::Int(*self.random.pick(&INTS)),
        }
    }

    /// A record whose fields may hold the records before it.
    pub fn record(&mut self) {
        let fields = (0..1 + self.random.below(3))
            .map(|index| (format!("f{index}"), self.ty(1)))
            .collect();
        let name = format!("R{}", self.records.len());
        self.records.push(Record { name, fields });
    }

    /// `count` statements; `nesting` bounds the depth of nested ifs and loops.
    fn block(&mut self, scope: &mut Scope, count: u64, nesting: u32) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        for _ in 0..count {
            let stmt = match self.random.below(11) {
                4..=6 => self.assign(scope),
                7 if nesting > 0 => Some(self.branch(scope, nesting)),
                8 if nesting > 0 => Some(self.repeat(scope, nesting)),
                9 if !scope.labels.is_empty() => Some(self.exit(scope)),
                10 if self.fallible => Some(self.assertion(scope)),
                _ => None,
            };
            stmts.push(stmt.unwrap_or_else(|| self.declare(scope)));
        }
        stmts
    }

    fn declare(&mut self, scope: &mut Scope) -> Stmt {
        let ty = self.ty(2);
        let value = self.expr(scope, &ty, 3);
        let name = self.fresh("l");
        scope.locals.push(Local {
            name: name.clone(),
            ty: ty.clone(),
            mutable: true,
        });
        Stmt::Let(name, ty, value)
    }

    fn assign(&mut self, scope: &Scope) -> Option<Stmt> {
        let targets = scope.targets(&self.records);
        if targets.is_empty() {
            return None;
        }
        let (name, projections, ty) =
            targets[self.random.below(targets.len() as u64) as usize].clone();
        let projections = projections
            .into_iter()
            .map(|projection| match projection {
                Projection::Index(_, length) => {
                    let int = *self.random.pick(&INTS);
                    Projection::Index(self.expr(scope, &Type::Int(int), 1), length)
                }
                field => field,
            })
            .collect();
        let value = self.expr(scope, &ty, 3);
        Some(Stmt::Assign(name, projections, value))
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
        let bound = 1 + self.random.below(4) as u32;
        let count = 1 + self.random.below(4);
        let body = self.block(&mut inner, count, nesting - 1);
        Stmt::For(label, counter, bound, body)
    }

    fn exit(&mut self, scope: &Scope) -> Stmt {
        let label = self.random.pick(&scope.labels).clone();
        let kind = *self.random.pick(&["break", "continue"]);
        let condition = self.expr(scope, &Type::Bool, 2);
        Stmt::Exit(condition, kind, label)
    }

    /// An assertion, mostly one that compares a value of the thread with a literal, so that it
    /// fails for some threads only.
    fn assertion(&mut self, scope: &Scope) -> Stmt {
        if self.random.chance(1, 4) {
            return Stmt::Assert(self.expr(scope, &Type::Bool, 2));
        }
        let int = *self.random.pick(&INTS);
        let value = self.expr(scope, &Type::Int(int), 2);
        let varying = self.varying(scope, int);
        let mixed = Expr::Binary(Box::new(value), "^", Box::new(varying));
        let comparison = self.random.pick(&["!=", "!=", "<", ">="]);
        let literal = Expr::Literal(int.literal(&mut self.random));
        Stmt::Assert(Expr::Binary(Box::new(mixed), comparison, Box::new(literal)))
    }

    /// A helper that the helpers after it and `run` may call; it calls only earlier ones.
    pub fn helper(&mut self) {
        let name = format!("h{}", self.helpers.len());
        let generic = self.random.chance(1, 4).then(|| "K".to_owned());
        let parameters = (0..1 + self.random.below(3))
            .map(|index| (format!("p{index}"), self.ty(1)))
            .collect::<Vec<_>>();
        let result = self.ty(1);
        let mut scope = Scope {
            generic: generic.clone(),
            varying: parameters
                .iter()
                .filter(|(_, ty)| matches!(ty, Type::Int(_)))
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

    /// `run(input) -> [u32; words]`, the code that every thread executes.
    pub fn run(&mut self, words: u32) -> (Function, Vec<Word>) {
        let mut scope = Scope {
            varying: vec!["input".into()],
            ..Scope::default()
        };
        scope.locals.push(Local {
            name: "input".into(),
            ty: U32,
            mutable: false,
        });
        let count = 3 + self.random.below(5);
        let body = self.block(&mut scope, count, 2);
        let mut values = Vec::new();
        let mut kinds = Vec::new();
        for _ in 0..words {
            let ty = self.ty(0);
            let value = Box::new(self.expr(&scope, &ty, 3));
            let (word, kind) = match ty {
                Type::Float => (Expr::Method(value, "to_bits", vec![]), Word::Float),
                _ => (Expr::Cast(value, U32), Word::Exact),
            };
            values.push(word);
            kinds.push(kind);
        }
        let result = Type::Array(Box::new(U32), words);
        let run = Function {
            name: "run".into(),
            generic: None,
            parameters: vec![("input".into(), U32)],
            result: result.clone(),
            body,
            value: Expr::Aggregate(result, values),
        };
        (run, kinds)
    }
}
