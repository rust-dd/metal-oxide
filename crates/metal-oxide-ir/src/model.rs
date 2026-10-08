use std::fmt;

use crate::{Access, AtomicOp, Scalar, Type, TypeTable};

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
    pub destination: Place,
    pub value: Expression,
    pub source: SourceLocation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Constant {
    Unit,
    Bool(bool),
    /// IEEE 754 bits, preserving NaNs and signed zero.
    F32(u32),
    F16(u16),
    U32(u32),
    I32(i32),
    U8(u8),
    U16(u16),
    I8(i8),
    I16(i16),
    Usize(u64),
}

impl Constant {
    pub fn ty(self) -> Type {
        match self {
            Self::Unit => Type::Unit,
            Self::Bool(_) => Type::Scalar(Scalar::Bool),
            Self::F32(_) => Type::Scalar(Scalar::F32),
            Self::F16(_) => Type::Scalar(Scalar::F16),
            Self::U32(_) => Type::Scalar(Scalar::U32),
            Self::I32(_) => Type::Scalar(Scalar::I32),
            Self::U8(_) => Type::Scalar(Scalar::U8),
            Self::U16(_) => Type::Scalar(Scalar::U16),
            Self::I8(_) => Type::Scalar(Scalar::I8),
            Self::I16(_) => Type::Scalar(Scalar::I16),
            Self::Usize(_) => Type::Scalar(Scalar::Usize),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Projection {
    Field(u32),
    /// The index is a pointer-width integer local.
    Index(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Place {
    pub local: usize,
    pub projection: Vec<Projection>,
}

impl Place {
    pub fn local(local: usize) -> Self {
        Self {
            local,
            projection: Vec::new(),
        }
    }

    pub fn contains(&self, other: &Self) -> bool {
        self.local == other.local && other.projection.starts_with(&self.projection)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operand {
    Place(Place),
    Constant(Constant),
    AggregateConstant { ty: Type, fields: Vec<Operand> },
}

impl Operand {
    pub fn local(local: usize) -> Self {
        Self::Place(Place::local(local))
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
pub enum MathOp {
    Abs,
    Min,
    Max,
    Sqrt,
    Fma,
}

impl MathOp {
    pub const fn arity(self) -> usize {
        match self {
            Self::Abs | Self::Sqrt => 1,
            Self::Min | Self::Max => 2,
            Self::Fma => 3,
        }
    }
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
    Bitcast(Operand, Scalar),
    Math {
        op: MathOp,
        arguments: Vec<Operand>,
    },
    Coordinates(Builtin),
    SimdCoordinate(SimdBuiltin),
    SimdSum(Operand),
    SimdShuffle {
        value: Operand,
        lane: Operand,
    },
    Dim3([Operand; 3]),
    Checked {
        scalar: Scalar,
        value: Operand,
        overflow: Operand,
    },
    Aggregate {
        ty: usize,
        fields: Vec<Operand>,
    },
    ThreadgroupAlloc {
        id: u32,
        element: Scalar,
        length: u32,
        access: Access,
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
    Atomic {
        op: AtomicOp,
        arguments: Vec<Operand>,
    },
    Call {
        function: usize,
        arguments: Vec<Operand>,
    },
}

impl Expression {
    pub fn operands(&self) -> Vec<&Operand> {
        match self {
            Self::Use(v)
            | Self::Unary(_, v)
            | Self::Cast(v, _)
            | Self::Bitcast(v, _)
            | Self::SimdSum(v) => vec![v],
            Self::SimdShuffle { value, lane } => vec![value, lane],
            Self::Binary(_, a, b) => vec![a, b],
            Self::Coordinates(_)
            | Self::SimdCoordinate(_)
            | Self::ThreadgroupAlloc { .. }
            | Self::ThreadgroupBarrier => {
                vec![]
            }
            Self::Dim3(values) => values.iter().collect(),
            Self::Checked {
                value, overflow, ..
            } => vec![value, overflow],
            Self::Aggregate { fields, .. } => fields.iter().collect(),
            Self::BufferLoad { buffer, index } => vec![buffer, index],
            Self::BufferStore {
                buffer,
                index,
                value,
            } => vec![buffer, index, value],
            Self::Call { arguments, .. }
            | Self::Math { arguments, .. }
            | Self::Atomic { arguments, .. } => arguments.iter().collect(),
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
