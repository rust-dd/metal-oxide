use super::types::{Record, Type, render as render_type};

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
    /// An index kept in bounds by a remainder of the array length.
    Index(Box<Expr>, Box<Expr>, u32),
    Aggregate(Type, Vec<Expr>),
}

#[derive(Clone, Debug)]
pub enum Projection {
    Field(String),
    Index(Expr, u32),
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let(String, Type, Expr),
    Assign(String, Vec<Projection>, Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    /// `for counter in 0..bound`, under a label that nested exits may name.
    For(String, String, u32, Vec<Stmt>),
    /// `if condition { break|continue 'label; }`
    Exit(Expr, &'static str, String),
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
            Expr::Index(array, index, length) => {
                format!(
                    "{}[({} as usize) % {length}]",
                    self.expr(array),
                    self.expr(index)
                )
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
        }
    }

    fn line(&mut self, indent: usize, text: &str) {
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
                Stmt::Assign(name, projections, value) => {
                    let mut place = name.clone();
                    for projection in projections {
                        match projection {
                            Projection::Field(field) => place = format!("{place}.{field}"),
                            Projection::Index(index, length) => {
                                place =
                                    format!("{place}[({} as usize) % {length}]", self.expr(index));
                            }
                        }
                    }
                    let value = self.expr(value);
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
                Stmt::For(label, counter, bound, body) => {
                    self.line(
                        indent,
                        &format!("'{label}: for {counter} in 0..{bound}_u32 {{"),
                    );
                    self.stmts(body, indent + 1);
                    self.line(indent, "}");
                }
                Stmt::Exit(condition, kind, label) => {
                    let condition = self.expr(condition);
                    self.line(indent, &format!("if {condition} {{ {kind} '{label}; }}"));
                }
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
