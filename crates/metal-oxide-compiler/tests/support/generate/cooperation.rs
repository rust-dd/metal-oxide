use super::{
    ast::{Expr, Function, Printer, Stmt},
    builder::Builder,
    model::COUNTERS,
    scope::{Local, Scope},
    types::{I32, Int, Type, U32, render as render_type},
};

const REDUCTIONS: [&str; 8] = [
    "sum",
    "min",
    "max",
    "and",
    "or",
    "xor",
    "inclusive_sum",
    "exclusive_sum",
];

const PERMUTATIONS: [&str; 4] = ["shuffle", "shuffle_up", "shuffle_down", "shuffle_xor"];

const ATOMICS: [&str; 7] = ["add", "sub", "min", "max", "and", "or", "xor"];

/// What the threads of a block compute together; each binds a mutable local.
#[derive(Clone, Debug)]
pub enum Cooperative {
    /// `simdgroup::op(value)`: a reduction, a scan or a vote.
    Collective {
        binding: String,
        ty: Type,
        op: &'static str,
        value: Expr,
    },
    /// `simdgroup::op(value, control & 31)`: a shuffle from the lane that `control` names, or by a
    /// uniform literal amount.
    Permute {
        binding: String,
        ty: Type,
        op: &'static str,
        value: Expr,
        control: Expr,
    },
    /// Each thread stores `value` at `lane ^ mask` of `buffer`, which holds an element per thread
    /// of the block; after a barrier, the binding reads the element at `index`.
    Exchange {
        buffer: String,
        binding: String,
        ty: Type,
        threads: u32,
        mask: u32,
        value: Expr,
        index: Expr,
    },
    /// The threads set the cells of `buffer` to `initial`, apply `op` with `value` at `index`, and
    /// after a barrier the binding reads the cell at `read`.
    Accumulate {
        buffer: String,
        binding: String,
        int: Int,
        cells: u32,
        op: &'static str,
        initial: String,
        index: Expr,
        value: Expr,
        read: Expr,
    },
    /// Every thread applies the program's one counter operation with `value` at `index` of the
    /// device counters, whose final words the host reads.
    Count {
        op: &'static str,
        index: Expr,
        value: Expr,
    },
}

/// What the statements of `cooperate` share: its parameters, which gain the buffers they use,
/// the threads per block, and the element type and operation of its device counters.
struct Phase {
    threads: u32,
    parameters: Vec<(String, Type)>,
    counters: Option<(Int, &'static str)>,
}

impl Cooperative {
    /// The expressions of the statement, in the order the printer writes them.
    pub fn operands_mut(&mut self) -> Vec<&mut Expr> {
        match self {
            Self::Collective { value, .. } => vec![value],
            Self::Permute { value, control, .. } => vec![value, control],
            Self::Exchange { value, index, .. } => vec![value, index],
            Self::Accumulate {
                index, value, read, ..
            } => vec![index, value, read],
            Self::Count { index, value, .. } => vec![index, value],
        }
    }
}

impl Printer<'_> {
    pub fn cooperative(&mut self, cooperative: &Cooperative, indent: usize) {
        let barrier = "unsafe { threadgroup::barrier() };";
        match cooperative {
            Cooperative::Collective {
                binding,
                ty,
                op,
                value,
            } => {
                let (ty, value) = (render_type(ty, self.records), self.expr(value));
                let text =
                    format!("let mut {binding}: {ty} = unsafe {{ simdgroup::{op}({value}) }};");
                self.line(indent, &text);
            }
            Cooperative::Permute {
                binding,
                ty,
                op,
                value,
                control,
            } => {
                let (ty, value) = (render_type(ty, self.records), self.expr(value));
                let control = self.expr(control);
                let text = format!(
                    "let mut {binding}: {ty} = unsafe {{ simdgroup::{op}({value}, {control} & 31) }};"
                );
                self.line(indent, &text);
            }
            Cooperative::Exchange {
                buffer,
                binding,
                ty,
                threads,
                mask,
                value,
                index,
            } => {
                let (ty, value) = (render_type(ty, self.records), self.expr(value));
                let index = self.expr(index);
                let store =
                    format!("unsafe {{ {buffer}.store_unchecked(lane ^ {mask}, {value}) }};");
                self.line(indent, &store);
                self.line(indent, barrier);
                let last = threads - 1;
                let load = format!(
                    "let mut {binding}: {ty} = unsafe {{ {buffer}.load_unchecked({index} & {last}) }};"
                );
                self.line(indent, &load);
                self.line(indent, barrier);
            }
            Cooperative::Accumulate {
                buffer,
                binding,
                int,
                cells,
                op,
                initial,
                index,
                value,
                read,
            } => {
                let (index, value, read) = (self.expr(index), self.expr(value), self.expr(read));
                let int = int.name();
                let set = format!(
                    "unsafe {{ {buffer}.store(lane % {cells}, {initial}, core::sync::atomic::Ordering::Relaxed) }};"
                );
                self.line(indent, &set);
                self.line(indent, barrier);
                let apply = format!(
                    "unsafe {{ {buffer}.fetch_{op}({index} % {cells}, {value}, core::sync::atomic::Ordering::Relaxed) }};"
                );
                self.line(indent, &apply);
                self.line(indent, barrier);
                let load = format!(
                    "let mut {binding}: {int} = unsafe {{ {buffer}.load({read} % {cells}, core::sync::atomic::Ordering::Relaxed) }};"
                );
                self.line(indent, &load);
                self.line(indent, barrier);
            }
            Cooperative::Count { op, index, value } => {
                let (index, value) = (self.expr(index), self.expr(value));
                let cells = COUNTERS.len();
                let count = format!(
                    "unsafe {{ counters.fetch_{op}({index} % {cells}, {value}, core::sync::atomic::Ordering::Relaxed) }};"
                );
                self.line(indent, &count);
            }
        }
    }
}

impl Builder {
    /// `cooperate(input, lane, buffers..) -> [u32; values]`, which the `threads` of a block run
    /// together. Checks before cooperative operations must be proved, so it calls nothing that
    /// may check.
    pub fn cooperation(&mut self, input: &Type, values: u32, threads: u32) -> Function {
        let fallible = std::mem::replace(&mut self.fallible, false);
        self.plain = true;
        let mut scope = Scope {
            varying: vec!["input".into(), "lane".into()],
            ..Scope::default()
        };
        let mut phase = Phase {
            threads,
            parameters: Vec::new(),
            counters: None,
        };
        for (name, ty) in [("input", input.clone()), ("lane", U32)] {
            scope.locals.push(Local {
                name: name.into(),
                ty: ty.clone(),
                mutable: false,
            });
            phase.parameters.push((name.to_owned(), ty));
        }
        if self.random.chance(1, 2) {
            let int = *self.random.pick(&[Int::U32, Int::I32]);
            let op = *self.random.pick(&ATOMICS);
            phase
                .parameters
                .push(("counters".into(), Type::Atomic(int)));
            phase.counters = Some((int, op));
        }
        let mut body = Vec::new();
        for _ in 0..2 + self.random.below(4) {
            let stmt = if self.random.chance(1, 4) {
                self.uniform_loop(&scope, &mut phase)
            } else {
                self.cooperative(&mut scope, &mut phase)
            };
            body.push(stmt);
        }
        let result = Type::Array(Box::new(U32), values);
        let words = (0..values)
            .map(|_| {
                let ty = self.ty(0);
                Expr::Cast(Box::new(self.expr(&scope, &ty, 3)), U32)
            })
            .collect::<Vec<_>>();
        self.fallible = fallible;
        self.plain = false;
        Function {
            name: "cooperate".into(),
            generic: None,
            parameters: phase.parameters,
            result: result.clone(),
            body,
            value: Expr::Aggregate(result, words),
        }
    }

    /// `for c in 0..bound { .. }` around cooperative statements and assignments to the locals
    /// before them; every thread runs the same iterations.
    fn uniform_loop(&mut self, scope: &Scope, phase: &mut Phase) -> Stmt {
        let label = self.fresh("o");
        let counter = self.fresh("c");
        let mut inner = scope.clone();
        inner.locals.push(Local {
            name: counter.clone(),
            ty: U32,
            mutable: false,
        });
        let mut body = Vec::new();
        for _ in 0..1 + self.random.below(3) {
            let assignment = self.random.chance(1, 3).then(|| self.assign(&inner));
            match assignment.flatten() {
                Some(assignment) => body.push(assignment),
                None => body.push(self.cooperative(&mut inner, phase)),
            }
        }
        let bound = 1 + self.random.below(3);
        let range = Expr::Literal(format!("0..{bound}_u32"));
        Stmt::For(label, counter, range, body)
    }

    /// A cooperative statement, whose binding the code after it can read.
    fn cooperative(&mut self, scope: &mut Scope, phase: &mut Phase) -> Stmt {
        let binding = self.fresh("l");
        let elements = [U32, I32, Type::Float];
        let integers = [U32, I32];
        let kinds = if phase.counters.is_some() { 6 } else { 5 };
        let (cooperative, ty) = match self.random.below(kinds) {
            0 => {
                let ty = self.random.pick(&integers).clone();
                let op = *self.random.pick(&REDUCTIONS);
                let value = self.expr(scope, &ty, 3);
                let collective = Cooperative::Collective {
                    binding: binding.clone(),
                    ty: ty.clone(),
                    op,
                    value,
                };
                (collective, ty)
            }
            1 => {
                let op = *self.random.pick(&["any", "all", "ballot"]);
                let ty = match op {
                    "ballot" => Type::Array(Box::new(U32), 2),
                    _ => Type::Bool,
                };
                let value = self.expr(scope, &Type::Bool, 3);
                let vote = Cooperative::Collective {
                    binding: binding.clone(),
                    ty: ty.clone(),
                    op,
                    value,
                };
                (vote, ty)
            }
            2 => {
                let ty = self.random.pick(&elements).clone();
                let op = *self.random.pick(&PERMUTATIONS);
                let value = self.expr(scope, &ty, 3);
                // Shifts and masks are uniform; each lane may read its own source.
                let control = match op {
                    "shuffle" => self.expr(scope, &U32, 2),
                    _ => Expr::Literal(format!("{}_u32", self.random.below(32))),
                };
                let permute = Cooperative::Permute {
                    binding: binding.clone(),
                    ty: ty.clone(),
                    op,
                    value,
                    control,
                };
                (permute, ty)
            }
            3 => {
                let ty = self.random.pick(&elements).clone();
                let buffer = self.fresh("s");
                let buffer_type = Type::Threadgroup(Box::new(ty.clone()), phase.threads);
                phase.parameters.push((buffer.clone(), buffer_type));
                let mask = self.random.below(u64::from(phase.threads)) as u32;
                let value = self.expr(scope, &ty, 3);
                let index = self.expr(scope, &U32, 2);
                let exchange = Cooperative::Exchange {
                    buffer,
                    binding: binding.clone(),
                    ty: ty.clone(),
                    threads: phase.threads,
                    mask,
                    value,
                    index,
                };
                (exchange, ty)
            }
            4 => {
                let int = *self.random.pick(&[Int::U32, Int::I32]);
                let buffer = self.fresh("a");
                let cells = 1 + self.random.below(4) as u32;
                let buffer_type = Type::AtomicThreadgroup(int, cells);
                phase.parameters.push((buffer.clone(), buffer_type));
                let op = *self.random.pick(&ATOMICS);
                let initial = int.literal(&mut self.random);
                let index = self.expr(scope, &U32, 2);
                let value = self.expr(scope, &Type::Int(int), 3);
                let read = self.expr(scope, &U32, 2);
                let accumulate = Cooperative::Accumulate {
                    buffer,
                    binding: binding.clone(),
                    int,
                    cells,
                    op,
                    initial,
                    index,
                    value,
                    read,
                };
                (accumulate, Type::Int(int))
            }
            _ => {
                let (int, op) = phase.counters.unwrap();
                let index = self.expr(scope, &U32, 2);
                let value = self.expr(scope, &Type::Int(int), 3);
                return Stmt::Cooperative(Cooperative::Count { op, index, value });
            }
        };
        scope.locals.push(Local {
            name: binding,
            ty,
            mutable: true,
        });
        Stmt::Cooperative(cooperative)
    }
}
