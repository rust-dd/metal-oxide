use super::{
    cooperation::Cooperative,
    types::{Record, Type, mutability, render as render_type},
};

#[derive(Clone, Debug)]
pub enum Expr {
    /// Rendered with its type suffix or path, as `7_u8` or `i16::MIN`.
    Literal(String),
    Local(String),
    Unary(&'static str, Box<Expr>),
    Binary(Box<Expr>, &'static str, Box<Expr>),
    Method(Box<Expr>, &'static str, Vec<Expr>),
    Cast(Box<Expr>, Type),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    /// An integer match on `scrutinee & 3`, with arms for 0, 1 and 2.
    Match(Box<Expr>, Box<[Expr; 4]>),
    /// A helper with its const argument, if it takes one.
    Call(String, Option<u32>, Vec<Expr>),
    Field(Box<Expr>, String),
    /// An index taken modulo the given value, if any; a modulus above the length, or none, can
    /// fail the bounds check.
    Index(Box<Expr>, Box<Expr>, Option<u32>),
    Aggregate(Type, Vec<Expr>),
    /// A variant constructor: `Some(value)`, `None::<T>` or `E0::V1(a, b)`.
    Variant(String, Vec<Expr>),
    /// A `match` whose patterns bind payloads that their arms use, as `Some(b3) => b3`.
    MatchVariant(Box<Expr>, Vec<(String, Expr)>),
    /// `matches!(value, pattern)`.
    Matches(Box<Expr>, String),
    /// `value?` in a helper that returns an `Option` or a `Result`.
    Try(Box<Expr>),
    /// `(*reference)`.
    Deref(Box<Expr>),
    /// `&place`, or `&mut place` when set.
    Borrow(Box<Expr>, bool),
    /// An element of a slice; the index is taken modulo the length when set.
    Element(Box<Expr>, Box<Expr>, bool),
    /// `&array[start..end]`, or `&mut array[start..end]` when set.
    Subslice(Box<Expr>, Bound, Bound, bool),
    /// `move |parameters| -> result { body }`.
    Closure(Vec<(String, Type)>, Type, Box<Expr>),
}

/// A subslice bound: a constant, or an index modulo the given value.
#[derive(Clone, Debug)]
pub enum Bound {
    Constant(u32),
    Varying(Box<Expr>, u32),
}

impl Expr {
    /// Whether the expression names the local `name`.
    pub fn mentions(&self, name: &str) -> bool {
        let any = |values: &[Expr]| values.iter().any(|value| value.mentions(name));
        match self {
            Expr::Literal(_) => false,
            Expr::Local(local) => local == name,
            Expr::Unary(_, value)
            | Expr::Cast(value, _)
            | Expr::Field(value, _)
            | Expr::Matches(value, _)
            | Expr::Try(value)
            | Expr::Deref(value)
            | Expr::Borrow(value, _) => value.mentions(name),
            Expr::Binary(left, _, right)
            | Expr::Index(left, right, _)
            | Expr::Element(left, right, _) => left.mentions(name) || right.mentions(name),
            Expr::Method(receiver, _, arguments) => receiver.mentions(name) || any(arguments),
            Expr::If(condition, then, otherwise) => {
                condition.mentions(name) || then.mentions(name) || otherwise.mentions(name)
            }
            Expr::Match(scrutinee, arms) => scrutinee.mentions(name) || any(&arms[..]),
            Expr::Call(callee, _, values) => callee == name || any(values),
            Expr::Aggregate(_, values) | Expr::Variant(_, values) => any(values),
            Expr::MatchVariant(scrutinee, arms) => {
                scrutinee.mentions(name) || arms.iter().any(|(_, arm)| arm.mentions(name))
            }
            Expr::Subslice(array, start, end, _) => {
                let bound = |bound: &Bound| matches!(bound, Bound::Varying(index, _) if index.mentions(name));
                array.mentions(name) || bound(start) || bound(end)
            }
            Expr::Closure(_, _, body) => body.mentions(name),
        }
    }
}

/// A step from a local to a place inside it; indices are taken modulo the length.
#[derive(Clone, Debug)]
pub enum Projection {
    Field(String),
    Index(u32),
    Deref,
    Element,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let(String, Type, Expr),
    Assign(Expr, Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    /// `for binding in iterable`, under a label that nested exits may name.
    For(String, String, Expr, Vec<Stmt>),
    /// `if condition { break|continue 'label; }`
    Exit(Expr, &'static str, String),
    Assert(Expr),
    /// A call whose result is unused, for the effects of its `&mut` arguments.
    Call(Expr),
    /// `let name = closure;`
    Closure(String, Expr),
    /// A statement of `cooperate`, which all threads of a block reach together.
    Cooperative(Cooperative),
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    /// The name of a `const` u32 parameter.
    pub generic: Option<String>,
    pub parameters: Vec<(String, Type)>,
    pub result: Type,
    pub body: Vec<Stmt>,
    pub value: Expr,
}

impl Function {
    /// Whether the function takes a `&mut` reference or slice; only statements call it, so that
    /// no other part of an expression can name what it borrows.
    pub fn mutates(&self) -> bool {
        self.parameters
            .iter()
            .any(|(_, ty)| matches!(ty, Type::Ref(_, true) | Type::Slice(_, true)))
    }
}

pub struct Printer<'a> {
    pub records: &'a [Record],
    pub text: String,
}

impl Printer<'_> {
    pub fn expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Literal(text) | Expr::Local(text) => text.clone(),
            Expr::Unary(operator, operand) => format!("({operator}{})", self.expr(operand)),
            Expr::Binary(left, operator, right) => {
                format!("({} {operator} {})", self.expr(left), self.expr(right))
            }
            Expr::Method(receiver, method, arguments) => {
                let arguments = arguments.iter().map(|a| self.expr(a)).collect::<Vec<_>>();
                format!("{}.{method}({})", self.expr(receiver), arguments.join(", "))
            }
            Expr::Cast(value, ty) => {
                format!(
                    "({} as {})",
                    self.expr(value),
                    render_type(ty, self.records)
                )
            }
            Expr::If(condition, then, otherwise) => format!(
                "(if {} {{ {} }} else {{ {} }})",
                self.expr(condition),
                self.expr(then),
                self.expr(otherwise)
            ),
            Expr::Match(scrutinee, arms) => format!(
                "(match {} & 3 {{ 0 => {}, 1 => {}, 2 => {}, _ => {} }})",
                self.expr(scrutinee),
                self.expr(&arms[0]),
                self.expr(&arms[1]),
                self.expr(&arms[2]),
                self.expr(&arms[3])
            ),
            Expr::Call(name, generic, arguments) => {
                let arguments = arguments.iter().map(|a| self.expr(a)).collect::<Vec<_>>();
                let generic = generic
                    .map(|value| format!("::<{value}>"))
                    .unwrap_or_default();
                format!("{name}{generic}({})", arguments.join(", "))
            }
            Expr::Field(value, field) => format!("{}.{field}", self.expr(value)),
            Expr::Index(array, index, Some(modulus)) => {
                let (array, index) = (self.expr(array), self.expr(index));
                format!("{array}[({index} as usize) % {modulus}]")
            }
            Expr::Index(array, index, None) => {
                format!("{}[{} as usize]", self.expr(array), self.expr(index))
            }
            Expr::Aggregate(ty, fields) => {
                let fields = fields.iter().map(|f| self.expr(f)).collect::<Vec<_>>();
                match ty {
                    Type::Array(..) => format!("[{}]", fields.join(", ")),
                    Type::Tuple(_) => format!("({},)", fields.join(", ")),
                    Type::Record(index) => {
                        let record = &self.records[*index];
                        let fields = record
                            .fields
                            .iter()
                            .zip(fields)
                            .map(|((name, _), value)| format!("{name}: {value}"))
                            .collect::<Vec<_>>();
                        format!("{} {{ {} }}", record.name, fields.join(", "))
                    }
                    _ => unreachable!("scalars are not aggregates"),
                }
            }
            Expr::Variant(name, payload) if payload.is_empty() => name.clone(),
            Expr::Variant(name, payload) => {
                let payload = payload
                    .iter()
                    .map(|value| self.expr(value))
                    .collect::<Vec<_>>();
                format!("{name}({})", payload.join(", "))
            }
            Expr::MatchVariant(scrutinee, arms) => {
                let arms = arms
                    .iter()
                    .map(|(pattern, arm)| format!("{pattern} => {}", self.expr(arm)))
                    .collect::<Vec<_>>();
                format!("(match {} {{ {} }})", self.expr(scrutinee), arms.join(", "))
            }
            Expr::Matches(value, pattern) => format!("matches!({}, {pattern})", self.expr(value)),
            Expr::Try(value) => format!("({}?)", self.expr(value)),
            Expr::Deref(reference) => format!("(*{})", self.expr(reference)),
            Expr::Borrow(place, mutable) => {
                format!("&{}{}", mutability(*mutable), self.expr(place))
            }
            Expr::Element(slice, index, true) => {
                let (slice, index) = (self.expr(slice), self.expr(index));
                format!("{slice}[({index} as usize) % {slice}.len()]")
            }
            Expr::Element(slice, index, false) => {
                format!("{}[{} as usize]", self.expr(slice), self.expr(index))
            }
            Expr::Closure(parameters, result, body) => {
                let parameters = parameters
                    .iter()
                    .map(|(name, ty)| format!("{name}: {}", render_type(ty, self.records)))
                    .collect::<Vec<_>>();
                let result = render_type(result, self.records);
                let body = self.expr(body);
                format!(
                    "(move |{}| -> {result} {{ {body} }})",
                    parameters.join(", ")
                )
            }
            Expr::Subslice(array, start, end, mutable) => format!(
                "&{}{}[{}..{}]",
                mutability(*mutable),
                self.expr(array),
                self.bound(start),
                self.bound(end)
            ),
        }
    }

    fn bound(&self, bound: &Bound) -> String {
        match bound {
            Bound::Constant(value) => value.to_string(),
            Bound::Varying(index, modulus) => {
                format!("(({} as usize) % {modulus})", self.expr(index))
            }
        }
    }

    pub fn line(&mut self, indent: usize, text: &str) {
        self.text.push_str(&"    ".repeat(indent));
        self.text.push_str(text);
        self.text.push('\n');
    }

    pub fn stmts(&mut self, stmts: &[Stmt], indent: usize) {
        for stmt in stmts {
            match stmt {
                Stmt::Let(name, ty, value) => {
                    let ty = render_type(ty, self.records);
                    let value = self.expr(value);
                    self.line(indent, &format!("let mut {name}: {ty} = {value};"));
                }
                Stmt::Assign(place, value) => {
                    let (place, value) = (self.expr(place), self.expr(value));
                    self.line(indent, &format!("{place} = {value};"));
                }
                Stmt::If(condition, then, otherwise) => {
                    let condition = self.expr(condition);
                    self.line(indent, &format!("if {condition} {{"));
                    self.stmts(then, indent + 1);
                    if otherwise.is_empty() {
                        self.line(indent, "}");
                    } else {
                        self.line(indent, "} else {");
                        self.stmts(otherwise, indent + 1);
                        self.line(indent, "}");
                    }
                }
                Stmt::For(label, binding, iterable, body) => {
                    let iterable = self.expr(iterable);
                    self.line(indent, &format!("'{label}: for {binding} in {iterable} {{"));
                    self.stmts(body, indent + 1);
                    self.line(indent, "}");
                }
                Stmt::Exit(condition, kind, label) => {
                    let condition = self.expr(condition);
                    self.line(indent, &format!("if {condition} {{ {kind} '{label}; }}"));
                }
                Stmt::Closure(name, closure) => {
                    let closure = self.expr(closure);
                    self.line(indent, &format!("let {name} = {closure};"));
                }
                Stmt::Call(call) => {
                    let call = self.expr(call);
                    self.line(indent, &format!("{call};"));
                }
                Stmt::Assert(condition) => {
                    let condition = self.expr(condition);
                    self.line(
                        indent,
                        &format!("assert!({condition}, \"generated assertion\");"),
                    );
                }
                Stmt::Cooperative(cooperative) => self.cooperative(cooperative, indent),
            }
        }
    }

    pub fn function(&mut self, function: &Function) {
        let generic = function
            .generic
            .as_ref()
            .map(|name| format!("<const {name}: u32>"))
            .unwrap_or_default();
        let parameters = function
            .parameters
            .iter()
            .map(|(name, ty)| format!("{name}: {}", render_type(ty, self.records)))
            .collect::<Vec<_>>();
        let result = render_type(&function.result, self.records);
        self.line(
            0,
            &format!(
                "fn {}{generic}({}) -> {result} {{",
                function.name,
                parameters.join(", ")
            ),
        );
        self.stmts(&function.body, 1);
        let value = self.expr(&function.value);
        self.line(1, &value);
        self.line(0, "}");
    }
}
