use std::fmt;

use crate::{Access, Aggregate, AtomicOp, Check, MatrixOp, Scalar, SimdOp, Type, TypeTable};

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
    /// A kernel's launch preconditions, which hold at its entry.
    pub requires: Vec<metal_oxide_artifact::Contract>,
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
    BF16(u16),
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    I8(i8),
    I16(i16),
    Usize(u64),
    Isize(i64),
}

impl Constant {
    pub fn ty(self) -> Type {
        match self {
            Self::Unit => Type::Unit,
            Self::Bool(_) => Type::Scalar(Scalar::Bool),
            Self::F32(_) => Type::Scalar(Scalar::F32),
            Self::F16(_) => Type::Scalar(Scalar::F16),
            Self::BF16(_) => Type::Scalar(Scalar::BF16),
            Self::U32(_) => Type::Scalar(Scalar::U32),
            Self::U64(_) => Type::Scalar(Scalar::U64),
            Self::I32(_) => Type::Scalar(Scalar::I32),
            Self::I64(_) => Type::Scalar(Scalar::I64),
            Self::U8(_) => Type::Scalar(Scalar::U8),
            Self::U16(_) => Type::Scalar(Scalar::U16),
            Self::I8(_) => Type::Scalar(Scalar::I8),
            Self::I16(_) => Type::Scalar(Scalar::I16),
            Self::Usize(_) => Type::Scalar(Scalar::Usize),
            Self::Isize(_) => Type::Scalar(Scalar::Isize),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Projection {
    Field(u32),
    /// The index is a pointer-width integer local. On a dereferenced slice it must be below the
    /// slice length; the bounds assertion before it establishes this.
    Index(usize),
    /// The value a reference points to, or the elements of a slice.
    Deref,
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

    /// Whether the place is reached through a pointer rather than stored in its local.
    pub fn is_indirect(&self) -> bool {
        self.projection.contains(&Projection::Deref)
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
    /// Integer or boolean ordering as an `i8`: -1, 0 or 1.
    Cmp,
    AddWithOverflow,
    SubWithOverflow,
    MulWithOverflow,
    /// The pointer moved by a `usize` or `isize` count of elements; it stays within or one past
    /// the values it points into.
    Offset,
    /// The element count from the second pointer to the first, which is not below it.
    OffsetFrom,
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
    CopySign,
    Sqrt,
    Fma,
    Ln,
    Exp,
    Sin,
    Cos,
    Powf,
    Tanh,
    Atan,
    Floor,
    Round,
    Ceil,
    Trunc,
    Exp2,
    Log2,
    Log10,
    Tan,
    Asin,
    Acos,
    /// `atan2(y, x)` with the arguments in that order.
    Atan2,
    Rsqrt,
    /// Returns the `(sin, cos)` tuple of one argument.
    SinCos,
}

impl MathOp {
    pub const fn arity(self) -> usize {
        match self {
            Self::Abs
            | Self::Sqrt
            | Self::Ln
            | Self::Exp
            | Self::Sin
            | Self::Cos
            | Self::SinCos
            | Self::Tanh
            | Self::Atan
            | Self::Floor
            | Self::Round
            | Self::Ceil
            | Self::Trunc
            | Self::Exp2
            | Self::Log2
            | Self::Log10
            | Self::Tan
            | Self::Asin
            | Self::Acos
            | Self::Rsqrt => 1,
            Self::Min | Self::Max | Self::CopySign | Self::Powf | Self::Atan2 => 2,
            Self::Fma => 3,
        }
    }
}

/// The interned `(f32, f32)` tuple that `MathOp::SinCos` produces.
pub fn sin_cos_type(types: &TypeTable) -> Option<Type> {
    let pair = Aggregate::Tuple(vec![Type::Scalar(Scalar::F32); 2]);
    types
        .iter()
        .find_map(|(id, shape)| (*shape == pair).then_some(Type::Aggregate(id)))
}

/// Integer operations with Rust semantics: counts are u32 and rotations wrap at the bit width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerOp {
    Min,
    Max,
    SaturatingAdd,
    SaturatingSub,
    LeadingZeros,
    TrailingZeros,
    CountOnes,
    ReverseBits,
    SwapBytes,
    RotateLeft,
    RotateRight,
}

impl IntegerOp {
    pub const fn arity(self) -> usize {
        match self {
            Self::LeadingZeros
            | Self::TrailingZeros
            | Self::CountOnes
            | Self::ReverseBits
            | Self::SwapBytes => 1,
            Self::Min
            | Self::Max
            | Self::SaturatingAdd
            | Self::SaturatingSub
            | Self::RotateLeft
            | Self::RotateRight => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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
    Select {
        condition: Operand,
        yes: Operand,
        no: Operand,
    },
    Cast(Operand, Scalar),
    /// f32 to integer without saturation; the value must be representable after truncation.
    UncheckedCast(Operand, Scalar),
    /// The bits of a value as a type of the same width: another scalar, or an integer and an
    /// array of integer lanes, whose first lane holds the low bits.
    Bitcast(Operand, Type),
    Math {
        op: MathOp,
        arguments: Vec<Operand>,
    },
    Integer {
        op: IntegerOp,
        arguments: Vec<Operand>,
    },
    Coordinates(Builtin),
    SimdCoordinate(SimdBuiltin),
    /// `block_idx.x * block_dim.x + thread_idx.x` as `usize`; a kernel that reads it accepts
    /// only one-dimensional launches.
    ThreadIndex,
    Simd {
        op: SimdOp,
        arguments: Vec<Operand>,
    },
    /// See [`MatrixOp`] for the operands.
    Matrix {
        op: MatrixOp,
        arguments: Vec<Operand>,
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
    /// An enum value of `variant` with its field values; other slots are unspecified.
    Variant {
        ty: usize,
        variant: u32,
        fields: Vec<Operand>,
    },
    ThreadgroupAlloc {
        id: u32,
        element: Scalar,
        length: u32,
        access: Access,
    },
    ThreadgroupBarrier,
    /// Thread-private array storage with no initial element writes.
    Uninit(Type),
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
        order: crate::AtomicOrdering,
        failure: crate::AtomicOrdering,
        arguments: Vec<Operand>,
    },
    /// The address of a place, which the operand names; the place stays in its address space.
    Ref {
        place: Operand,
        mutable: bool,
    },
    /// The length of a slice.
    Len(Operand),
    /// A slice of a whole array behind a reference, or of a kernel buffer parameter.
    Slice(Operand),
    /// `slice[start..end]`; the bounds assertions before it establish `start <= end <= len`.
    Subslice {
        slice: Operand,
        start: Operand,
        end: Operand,
    },
    /// `length` elements from the thin pointer `data`, as a slice.
    SliceFromParts {
        data: Operand,
        length: Operand,
    },
    /// The pointer as this pointer type, which views the same values in the same space with
    /// any mutability; a slice converts to its data pointer.
    PointerCast(Operand, Type),
    Call {
        function: usize,
        arguments: Vec<Operand>,
    },
    /// A device log message; `format` keeps Rust's `{}` placeholders.
    Log {
        format: String,
        arguments: Vec<Operand>,
    },
}

impl Expression {
    pub fn operands(&self) -> Vec<&Operand> {
        match self {
            Self::Use(v)
            | Self::Unary(_, v)
            | Self::Cast(v, _)
            | Self::UncheckedCast(v, _)
            | Self::Bitcast(v, _)
            | Self::Ref { place: v, .. }
            | Self::Len(v)
            | Self::Slice(v)
            | Self::PointerCast(v, _) => vec![v],
            Self::Binary(_, a, b) => vec![a, b],
            Self::Select { condition, yes, no } => vec![condition, yes, no],
            Self::Coordinates(_)
            | Self::SimdCoordinate(_)
            | Self::ThreadIndex
            | Self::ThreadgroupAlloc { .. }
            | Self::ThreadgroupBarrier
            | Self::Uninit(_) => {
                vec![]
            }
            Self::Dim3(values) => values.iter().collect(),
            Self::Checked {
                value, overflow, ..
            } => vec![value, overflow],
            Self::Aggregate { fields, .. } | Self::Variant { fields, .. } => {
                fields.iter().collect()
            }
            Self::BufferLoad { buffer, index } => vec![buffer, index],
            Self::BufferStore {
                buffer,
                index,
                value,
            } => vec![buffer, index, value],
            Self::Subslice { slice, start, end } => vec![slice, start, end],
            Self::SliceFromParts { data, length } => vec![data, length],
            Self::Call { arguments, .. }
            | Self::Log { arguments, .. }
            | Self::Math { arguments, .. }
            | Self::Integer { arguments, .. }
            | Self::Simd { arguments, .. }
            | Self::Matrix { arguments, .. }
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
        check: Check,
    },
    /// An explicit panic with its literal message.
    Panic(String),
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
            Self::Panic(_) | Self::Return | Self::Unreachable => vec![],
        }
    }

    pub fn operands(&self) -> Vec<&Operand> {
        match self {
            Self::Branch { condition, .. }
            | Self::Switch {
                discriminant: condition,
                ..
            }
            | Self::Assert { condition, .. } => vec![condition],
            Self::Goto(_) | Self::Panic(_) | Self::Return | Self::Unreachable => vec![],
        }
    }
}
