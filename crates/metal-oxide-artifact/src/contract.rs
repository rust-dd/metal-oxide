use serde::{Deserialize, Serialize};

use crate::Error;

/// A relation from `#[kernel(requires(...))]` that a launch checks before it encodes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Contract {
    /// The relation as written in the kernel source.
    pub relation: String,
    pub left: Term,
    pub comparison: Comparison,
    pub right: Term,
}

/// An integer over a launch's arguments.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Term {
    /// The element count of the buffer parameter at this index.
    Length(usize),
    /// The integer value parameter at this index.
    Value(usize),
    Literal(u64),
    Add(Box<Term>, Box<Term>),
    Sub(Box<Term>, Box<Term>),
    Mul(Box<Term>, Box<Term>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Comparison {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// What a parameter name stands for in a relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    /// A buffer, whose `len()` a relation reads.
    Buffer,
    /// An integer value.
    Integer,
}

impl Contract {
    /// Parses `name.len()` of a buffer, an integer parameter `name` or a literal, combined with
    /// `+`, `-`, `*` and parentheses, on both sides of one comparison.
    pub fn parse(
        relation: &str,
        parameter: impl Fn(&str) -> Option<(usize, Operand)>,
    ) -> Result<Self, Error> {
        let mut parser = Parser {
            tokens: tokens(relation)?,
            position: 0,
            parameter: &parameter,
        };
        let left = parser.sum()?;
        let Some(Token::Comparison(comparison)) = parser.next() else {
            return Err(Error(format!("{relation}: expected one comparison")));
        };
        let right = parser.sum()?;
        if parser.position != parser.tokens.len() {
            return Err(Error(format!("{relation}: unexpected trailing tokens")));
        }
        Ok(Self {
            relation: relation.to_owned(),
            left,
            comparison,
            right,
        })
    }

    /// Whether the relation holds for a launch; arithmetic that leaves `i128` fails it.
    pub fn holds(&self, length: impl Fn(usize) -> u64, value: impl Fn(usize) -> i128) -> bool {
        let evaluate = |term: &Term| term.evaluate(&length, &value);
        let (Some(left), Some(right)) = (evaluate(&self.left), evaluate(&self.right)) else {
            return false;
        };
        match self.comparison {
            Comparison::Eq => left == right,
            Comparison::Ne => left != right,
            Comparison::Lt => left < right,
            Comparison::Le => left <= right,
            Comparison::Gt => left > right,
            Comparison::Ge => left >= right,
        }
    }
}

impl Term {
    fn evaluate(
        &self,
        length: &impl Fn(usize) -> u64,
        value: &impl Fn(usize) -> i128,
    ) -> Option<i128> {
        Some(match self {
            Self::Length(index) => i128::from(length(*index)),
            Self::Value(index) => value(*index),
            Self::Literal(literal) => i128::from(*literal),
            Self::Add(a, b) => a
                .evaluate(length, value)?
                .checked_add(b.evaluate(length, value)?)?,
            Self::Sub(a, b) => a
                .evaluate(length, value)?
                .checked_sub(b.evaluate(length, value)?)?,
            Self::Mul(a, b) => a
                .evaluate(length, value)?
                .checked_mul(b.evaluate(length, value)?)?,
        })
    }

    /// The parameters the term reads, with what it reads of each.
    pub fn parameters(&self, found: &mut Vec<(usize, Operand)>) {
        match self {
            Self::Length(index) => found.push((*index, Operand::Buffer)),
            Self::Value(index) => found.push((*index, Operand::Integer)),
            Self::Literal(_) => {}
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) => {
                a.parameters(found);
                b.parameters(found);
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Name(String),
    Literal(u64),
    Length,
    Open,
    Close,
    Plus,
    Minus,
    Star,
    Comparison(Comparison),
}

fn tokens(relation: &str) -> Result<Vec<Token>, Error> {
    let error = || Error(format!("{relation}: unsupported relation syntax"));
    let bytes = relation.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        let rest = &relation[index..];
        let byte = bytes[index];
        let (token, width) = if byte.is_ascii_whitespace() {
            index += 1;
            continue;
        } else if let Some(stripped) = rest.strip_prefix(".len()") {
            (Token::Length, rest.len() - stripped.len())
        } else if byte.is_ascii_digit() {
            let width = rest
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .unwrap_or(rest.len());
            (
                Token::Literal(literal(&rest[..width]).ok_or_else(error)?),
                width,
            )
        } else if byte.is_ascii_alphabetic() || byte == b'_' {
            let width = rest
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .unwrap_or(rest.len());
            (Token::Name(rest[..width].to_owned()), width)
        } else {
            let comparison = [
                ("==", Comparison::Eq),
                ("!=", Comparison::Ne),
                ("<=", Comparison::Le),
                (">=", Comparison::Ge),
                ("<", Comparison::Lt),
                (">", Comparison::Gt),
            ]
            .into_iter()
            .find(|(text, _)| rest.starts_with(text));
            match (comparison, byte) {
                (Some((text, comparison)), _) => (Token::Comparison(comparison), text.len()),
                (None, b'(') => (Token::Open, 1),
                (None, b')') => (Token::Close, 1),
                (None, b'+') => (Token::Plus, 1),
                (None, b'-') => (Token::Minus, 1),
                (None, b'*') => (Token::Star, 1),
                _ => return Err(error()),
            }
        };
        tokens.push(token);
        index += width;
    }
    Ok(tokens)
}

/// A decimal integer literal with `_` separators and an optional integer type suffix.
fn literal(text: &str) -> Option<u64> {
    let (digits, suffix) = text.split_at(
        text.find(|c: char| c.is_ascii_alphabetic())
            .unwrap_or(text.len()),
    );
    let types = [
        "", "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize",
    ];
    if !types.contains(&suffix) {
        return None;
    }
    digits.replace('_', "").parse().ok()
}

struct Parser<'a, P> {
    tokens: Vec<Token>,
    position: usize,
    parameter: &'a P,
}

impl<P: Fn(&str) -> Option<(usize, Operand)>> Parser<'_, P> {
    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.position).cloned();
        self.position += usize::from(token.is_some());
        token
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn sum(&mut self) -> Result<Term, Error> {
        let mut term = self.product()?;
        loop {
            let combine = match self.peek() {
                Some(Token::Plus) => Term::Add,
                Some(Token::Minus) => Term::Sub,
                _ => return Ok(term),
            };
            self.position += 1;
            term = combine(Box::new(term), Box::new(self.product()?));
        }
    }

    fn product(&mut self) -> Result<Term, Error> {
        let mut term = self.atom()?;
        while self.peek() == Some(&Token::Star) {
            self.position += 1;
            term = Term::Mul(Box::new(term), Box::new(self.atom()?));
        }
        Ok(term)
    }

    fn atom(&mut self) -> Result<Term, Error> {
        match self.next() {
            Some(Token::Literal(literal)) => Ok(Term::Literal(literal)),
            Some(Token::Open) => {
                let term = self.sum()?;
                match self.next() {
                    Some(Token::Close) => Ok(term),
                    _ => Err(Error("unbalanced parentheses in a relation".into())),
                }
            }
            Some(Token::Name(name)) => {
                let length = self.peek() == Some(&Token::Length);
                self.position += usize::from(length);
                match ((self.parameter)(&name), length) {
                    (Some((index, Operand::Buffer)), true) => Ok(Term::Length(index)),
                    (Some((index, Operand::Integer)), false) => Ok(Term::Value(index)),
                    _ if length => {
                        Err(Error(format!("`{name}.len()` requires a buffer parameter")))
                    }
                    _ => Err(Error(format!("`{name}` is not an integer parameter"))),
                }
            }
            _ => Err(Error(
                "a relation compares lengths, integer parameters and literals".into(),
            )),
        }
    }
}
