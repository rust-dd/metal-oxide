use std::fmt;

use crate::{Scalar, Type, TypeTable};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub source: SourceLocation,
    pub message: String,
}

impl Error {
    pub fn new(source: &SourceLocation, message: impl Into<String>) -> Self {
        Self {
            source: source.clone(),
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}: {}",
            self.source.file, self.source.line, self.source.column, self.message
        )
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Module {
    pub functions: Vec<Function>,
    pub types: TypeTable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub kernel: bool,
    pub required_block: Option<[u32; 3]>,
    pub parameters: usize,
    /// Local zero is the return place; parameters start at local one.
    pub locals: Vec<Type>,
    pub blocks: Vec<Block>,
    pub source: SourceLocation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
    pub source: SourceLocation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    pub destination: usize,
    pub value: Expression,
    pub source: SourceLocation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Constant {
    Unit,
    Bool(bool),
    /// IEEE 754 bits, preserving NaNs and signed zero.
    F32(u32),
    U32(u32),
    I32(i32),
    U8(u8),
    U16(u16),
}

impl Constant {
    pub fn ty(self) -> Type {
        match self {
            Self::Unit => Type::Unit,
            Self::Bool(_) => Type::Scalar(Scalar::Bool),
            Self::F32(_) => Type::Scalar(Scalar::F32),
            Self::U32(_) => Type::Scalar(Scalar::U32),
            Self::I32(_) => Type::Scalar(Scalar::I32),
            Self::U8(_) => Type::Scalar(Scalar::U8),
            Self::U16(_) => Type::Scalar(Scalar::U16),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operand {
    Place { local: usize, field: Option<u32> },
    Constant(Constant),
}

impl Operand {
    pub fn local(local: usize) -> Self {
        Self::Place { local, field: None }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    AddWithOverflow,
    SubWithOverflow,
    MulWithOverflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Builtin {
    ThreadIdx,
    BlockIdx,
    BlockDim,
    GridDim,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdBuiltin {
    Lane,
    Size,
    Group,
    Count,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expression {
    Use(Operand),
    Binary(BinaryOp, Operand, Operand),
    Unary(UnaryOp, Operand),
    Cast(Operand, Scalar),
    Coordinates(Builtin),
    SimdCoordinate(SimdBuiltin),
    SimdSum(Operand),
    SimdShuffle {
        value: Operand,
        lane: Operand,
    },
    Dim3([Operand; 3]),
    Aggregate {
        ty: usize,
        fields: Vec<Operand>,
    },
    AggregateUpdate {
        aggregate: Operand,
        field: u32,
        value: Operand,
    },
    ThreadgroupAlloc {
        id: u32,
        element: Scalar,
        length: u32,
    },
    ThreadgroupBarrier,
    BufferLoad {
        buffer: Operand,
        index: Operand,
    },
    BufferStore {
        buffer: Operand,
        index: Operand,
        value: Operand,
    },
    AtomicAdd {
        buffer: Operand,
        index: Operand,
        value: Operand,
    },
    Call {
        function: usize,
        arguments: Vec<Operand>,
    },
}

impl Expression {
    pub fn operands(&self) -> Vec<&Operand> {
        match self {
            Self::Use(v) | Self::Unary(_, v) | Self::Cast(v, _) | Self::SimdSum(v) => vec![v],
            Self::SimdShuffle { value, lane } => vec![value, lane],
            Self::Binary(_, a, b) => vec![a, b],
            Self::Coordinates(_)
            | Self::SimdCoordinate(_)
            | Self::ThreadgroupAlloc { .. }
            | Self::ThreadgroupBarrier => {
                vec![]
            }
            Self::Dim3(values) => values.iter().collect(),
            Self::Aggregate { fields, .. } => fields.iter().collect(),
            Self::AggregateUpdate {
                aggregate, value, ..
            } => vec![aggregate, value],
            Self::BufferLoad { buffer, index } => vec![buffer, index],
            Self::BufferStore {
                buffer,
                index,
                value,
            }
            | Self::AtomicAdd {
                buffer,
                index,
                value,
            } => vec![buffer, index, value],
            Self::Call { arguments, .. } => arguments.iter().collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Terminator {
    Goto(usize),
    Branch {
        condition: Operand,
        then_block: usize,
        else_block: usize,
    },
    Switch {
        discriminant: Operand,
        cases: Vec<(Constant, usize)>,
        otherwise: usize,
    },
    Assert {
        condition: Operand,
        expected: bool,
        enabled: bool,
        target: usize,
        message: String,
    },
    Return,
    Unreachable,
}

impl Terminator {
    pub fn successors(&self) -> Vec<usize> {
        match self {
            Self::Goto(target) | Self::Assert { target, .. } => vec![*target],
            Self::Branch {
                then_block,
                else_block,
                ..
            } => vec![*then_block, *else_block],
            Self::Switch {
                cases, otherwise, ..
            } => {
                let mut targets = Vec::new();
                for target in cases.iter().map(|(_, target)| target).chain([otherwise]) {
                    if !targets.contains(target) {
                        targets.push(*target);
                    }
                }
                targets
            }
            Self::Return | Self::Unreachable => vec![],
        }
    }
}
