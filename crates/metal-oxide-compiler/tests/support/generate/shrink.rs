use super::{
    Program,
    ast::{Bound, Expr, Function, Stmt},
    types::Type,
};

impl Program {
    /// The program that single simplifications reach while `fails` holds: fewer helpers and
    /// statements, flatter control flow and simpler expressions. `fails` holds for `self`.
    pub fn shrink(self, mut fails: impl FnMut(&Program) -> bool) -> Program {
        let mut program = self;
        let mut start = 0;
        loop {
            let candidates = program.simplifications();
            let count = candidates.len();
            let next = (start..count)
                .chain(0..start.min(count))
                .find(|&index| fails(&candidates[index]));
            match next {
                Some(index) => {
                    program = candidates.into_iter().nth(index).unwrap();
                    start = index;
                }
                None => return program,
            }
        }
    }

    fn simplifications(&self) -> Vec<Program> {
        let mut candidates = Vec::new();
        for index in (0..self.helpers.len()).rev() {
            let mut program = self.clone();
            program.helpers.remove(index);
            candidates.push(program);
        }
        for function in 0..self.functions() {
            for body in removals(&self.function(function).body) {
                let mut program = self.clone();
                program.function_mut(function).body = body;
                candidates.push(program);
            }
        }
        for function in 0..self.functions() {
            let mut expressions = Vec::new();
            each_expression(&mut self.function(function).clone(), &mut |expr| {
                expressions.push(simpler(expr));
                true
            });
            for (index, replacements) in expressions.into_iter().enumerate() {
                for replacement in replacements {
                    let mut program = self.clone();
                    let mut position = 0;
                    let mut replacement = Some(replacement);
                    each_expression(program.function_mut(function), &mut |expr| {
                        if position == index {
                            *expr = replacement.take().unwrap();
                            return false;
                        }
                        position += 1;
                        true
                    });
                    candidates.push(program);
                }
            }
        }
        candidates
    }

    /// The number of functions: the helpers, `cooperate` if there is one, then `run`.
    fn functions(&self) -> usize {
        self.helpers.len() + usize::from(self.cooperation.is_some()) + 1
    }

    fn function(&self, index: usize) -> &Function {
        let functions = self.helpers.iter().chain(&self.cooperation);
        functions.chain([&self.run]).nth(index).unwrap()
    }

    fn function_mut(&mut self, index: usize) -> &mut Function {
        let functions = self.helpers.iter_mut().chain(&mut self.cooperation);
        functions.chain([&mut self.run]).nth(index).unwrap()
    }
}

/// `stmts` without one statement, with a branch or a loop body in place of its statement, or
/// with such a change in a nested block.
fn removals(stmts: &[Stmt]) -> Vec<Vec<Stmt>> {
    let mut found = Vec::new();
    for index in (0..stmts.len()).rev() {
        let with = |replacement: &[Stmt]| {
            let mut copy = stmts.to_vec();
            copy.splice(index..=index, replacement.iter().cloned());
            copy
        };
        found.push(with(&[]));
        match &stmts[index] {
            Stmt::If(condition, then, otherwise) => {
                found.push(with(then));
                found.push(with(otherwise));
                for then in removals(then) {
                    found.push(with(&[Stmt::If(
                        condition.clone(),
                        then,
                        otherwise.clone(),
                    )]));
                }
                for otherwise in removals(otherwise) {
                    found.push(with(&[Stmt::If(
                        condition.clone(),
                        then.clone(),
                        otherwise,
                    )]));
                }
            }
            Stmt::For(label, binding, iterable, body) => {
                found.push(with(body));
                for body in removals(body) {
                    let stmt = Stmt::For(label.clone(), binding.clone(), iterable.clone(), body);
                    found.push(with(&[stmt]));
                }
            }
            Stmt::While(label, counter, bound, condition, body) => {
                for body in removals(body) {
                    let (label, counter) = (label.clone(), counter.clone());
                    let stmt = Stmt::While(label, counter, bound.clone(), condition.clone(), body);
                    found.push(with(&[stmt]));
                }
            }
            _ => {}
        }
    }
    found
}

/// Calls `visit` on the function's expressions in pre-order until it returns false. Places
/// that statements assign or borrow are not visited: no other expression can stand for them.
fn each_expression(function: &mut Function, visit: &mut dyn FnMut(&mut Expr) -> bool) {
    if statements(&mut function.body, visit) {
        expression(&mut function.value, visit);
    }
}

fn statements(stmts: &mut [Stmt], visit: &mut dyn FnMut(&mut Expr) -> bool) -> bool {
    stmts.iter_mut().all(|stmt| match stmt {
        Stmt::Let(_, _, value)
        | Stmt::Assign(_, value)
        | Stmt::Exit(value, _, _)
        | Stmt::Assert(value)
        | Stmt::Return(value)
        | Stmt::Call(value)
        | Stmt::Closure(_, value) => expression(value, visit),
        Stmt::If(condition, then, otherwise) => {
            expression(condition, visit) && statements(then, visit) && statements(otherwise, visit)
        }
        Stmt::For(_, _, iterable, body) => expression(iterable, visit) && statements(body, visit),
        Stmt::While(_, _, bound, condition, body) => {
            expression(bound, visit) && expression(condition, visit) && statements(body, visit)
        }
        Stmt::Cooperative(cooperative) => cooperative
            .operands_mut()
            .into_iter()
            .all(|operand| expression(operand, visit)),
    })
}

fn expression(expr: &mut Expr, visit: &mut dyn FnMut(&mut Expr) -> bool) -> bool {
    visit(expr)
        && children(expr)
            .into_iter()
            .all(|child| expression(child, visit))
}

/// The operands of `expr` that hold values rather than places.
fn children(expr: &mut Expr) -> Vec<&mut Expr> {
    match expr {
        Expr::Literal(_) | Expr::Local(_) | Expr::Field(..) | Expr::Deref(_) | Expr::Borrow(..) => {
            Vec::new()
        }
        Expr::Unary(_, value)
        | Expr::Cast(value, _)
        | Expr::Matches(value, _)
        | Expr::Try(value)
        | Expr::Closure(_, _, value) => vec![value],
        Expr::Index(_, index, _) | Expr::Element(_, index, _) => vec![index],
        Expr::Binary(left, _, right) | Expr::Extreme(_, left, right) => vec![left, right],
        Expr::Method(receiver, _, arguments) => {
            std::iter::once(&mut **receiver).chain(arguments).collect()
        }
        Expr::If(condition, then, otherwise) => vec![condition, then, otherwise],
        Expr::Match(scrutinee, arms) => std::iter::once(&mut **scrutinee)
            .chain(arms.iter_mut())
            .collect(),
        Expr::MatchVariant(scrutinee, arms) => std::iter::once(&mut **scrutinee)
            .chain(arms.iter_mut().map(|(_, arm)| arm))
            .collect(),
        Expr::Call(_, _, values) | Expr::Aggregate(_, values) | Expr::Variant(_, values) => {
            values.iter_mut().collect()
        }
        Expr::Subslice(_, start, end, _) => [start, end]
            .into_iter()
            .filter_map(|bound| match bound {
                Bound::Varying(index, _) => Some(&mut **index),
                Bound::Constant(_) => None,
            })
            .collect(),
    }
}

/// Simpler expressions that may stand for `expr`: its branches or operands, a boolean in
/// place of a comparison, and zeros in place of casts and literals. Those of another type
/// make the program fail to build, which rejects them.
fn simpler(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::If(_, then, otherwise) => vec![(**then).clone(), (**otherwise).clone()],
        Expr::Match(_, arms) => arms.to_vec(),
        Expr::MatchVariant(_, arms) => arms.iter().map(|(_, arm)| arm.clone()).collect(),
        Expr::Binary(left, operator, right) => match *operator {
            "==" | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||" => {
                vec![literal("true"), literal("false")]
            }
            _ => vec![(**left).clone(), (**right).clone()],
        },
        Expr::Extreme(_, left, right) => vec![(**left).clone(), (**right).clone()],
        Expr::Matches(..) => vec![literal("true"), literal("false")],
        Expr::Unary(_, operand) => vec![(**operand).clone()],
        Expr::Method(receiver, _, arguments) => std::iter::once(&**receiver)
            .chain(arguments)
            .cloned()
            .collect(),
        Expr::Call(_, _, arguments) => arguments.clone(),
        Expr::Cast(_, Type::Int(int)) => vec![literal(&format!("0_{}", int.name()))],
        Expr::Cast(_, Type::Float) => vec![literal("0.0_f32")],
        Expr::Literal(text) => zero(text)
            .filter(|zero| zero != text)
            .map(|zero| vec![literal(&zero)])
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// The zero of a numeric literal's type, as `0_u8` for `i8::MAX` or `0.0_f32` for `(-1.5_f32)`.
fn zero(text: &str) -> Option<String> {
    let ty = match text.split_once("::") {
        Some((ty, _)) => ty,
        None => text.trim_end_matches(')').rsplit_once('_')?.1,
    };
    match ty {
        "f32" => Some("0.0_f32".into()),
        "u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize" => {
            Some(format!("0_{ty}"))
        }
        _ => None,
    }
}

fn literal(text: &str) -> Expr {
    Expr::Literal(text.into())
}
