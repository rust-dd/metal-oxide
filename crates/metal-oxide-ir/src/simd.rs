use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdOp {
    Sum,
    Min,
    Max,
    And,
    Or,
    Xor,
    InclusiveSum,
    ExclusiveSum,
    Shuffle,
    ShuffleUp,
    ShuffleDown,
    ShuffleXor,
    Any,
    All,
    Ballot,
}
impl SimdOp {
    pub const fn arity(self) -> usize {
        if matches!(
            self,
            Self::Shuffle | Self::ShuffleUp | Self::ShuffleDown | Self::ShuffleXor
        ) {
            2
        } else {
            1
        }
    }
    pub const fn uniform_control(self) -> bool {
        matches!(self, Self::ShuffleUp | Self::ShuffleDown | Self::ShuffleXor)
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Min => "min",
            Self::Max => "max",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::InclusiveSum => "inclusive_sum",
            Self::ExclusiveSum => "exclusive_sum",
            Self::Shuffle => "shuffle",
            Self::ShuffleUp => "shuffle_up",
            Self::ShuffleDown => "shuffle_down",
            Self::ShuffleXor => "shuffle_xor",
            Self::Any => "any",
            Self::All => "all",
            Self::Ballot => "ballot",
        }
    }
}

pub fn ballot_type(types: &TypeTable) -> Option<Type> {
    types.iter().find_map(|(id, shape)| {
        matches!(
            shape,
            Aggregate::Array {
                element: Type::Scalar(Scalar::U32),
                length: 2
            }
        )
        .then_some(Type::Aggregate(id))
    })
}

pub(crate) fn simd_type(
    module: &Module,
    function: &Function,
    op: SimdOp,
    operands: &[Operand],
    source: &SourceLocation,
) -> Result<Type, Error> {
    if operands.len() != op.arity() {
        return Err(Error::new(source, "invalid SIMD argument count"));
    }
    let ty = operand_type(module, function, &operands[0], source)?;
    if matches!(op, SimdOp::Any | SimdOp::All | SimdOp::Ballot) {
        if ty != Type::Scalar(Scalar::Bool) {
            return Err(Error::new(source, "SIMD vote requires bool"));
        }
        return if op == SimdOp::Ballot {
            ballot_type(&module.types)
                .ok_or_else(|| Error::new(source, "SIMD ballot requires a two-word u32 result"))
        } else {
            Ok(ty)
        };
    }
    if !matches!(ty, Type::Scalar(Scalar::F32 | Scalar::U32 | Scalar::I32)) {
        return Err(Error::new(
            source,
            "SIMD numeric operations require f32/u32/i32",
        ));
    }
    if matches!(
        op,
        SimdOp::And | SimdOp::Or | SimdOp::Xor | SimdOp::ShuffleXor
    ) && ty == Type::Scalar(Scalar::F32)
    {
        return Err(Error::new(
            source,
            "integer SIMD operation requires u32/i32",
        ));
    }
    if op.arity() == 2 {
        if operand_type(module, function, &operands[1], source)? != Type::Scalar(Scalar::U32) {
            return Err(Error::new(source, "SIMD source control requires u32"));
        }
        if matches!(operands[1], Operand::Constant(Constant::U32(value)) if value >= 64) {
            return Err(Error::new(
                source,
                "invalid static SIMD source control: maximum SIMD width is 64",
            ));
        }
    }
    Ok(ty)
}
