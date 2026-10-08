use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicOp {
    Load,
    Store,
    Exchange,
    CompareExchangeWeak,
    Add,
    Sub,
    Min,
    Max,
    And,
    Or,
    Xor,
}
impl AtomicOp {
    pub const fn arity(self) -> usize {
        match self {
            Self::Load => 0,
            Self::CompareExchangeWeak => 2,
            _ => 1,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::Store => "store",
            Self::Exchange => "exchange",
            Self::CompareExchangeWeak => "compare_exchange_weak",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Min => "min",
            Self::Max => "max",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
        }
    }
}
pub(crate) fn atomic_type(
    module: &Module,
    function: &Function,
    op: AtomicOp,
    operands: &[Operand],
    source: &SourceLocation,
) -> Result<Type, Error> {
    if operands.len() != 2 + op.arity() {
        return Err(Error::new(source, "invalid atomic argument count"));
    }
    let ty = |value| operand_type(module, function, value, source);
    let Type::Buffer {
        element: Element::Scalar(scalar @ (Scalar::U32 | Scalar::I32)),
        access: Access::Atomic,
        ..
    } = ty(&operands[0])?
    else {
        return Err(Error::new(
            source,
            "atomic operations require an i32/u32 atomic buffer",
        ));
    };
    if ty(&operands[1])? != Type::Scalar(Scalar::U32) {
        return Err(Error::new(source, "atomic index requires u32"));
    }
    for value in &operands[2..] {
        if ty(value)? != Type::Scalar(scalar) {
            return Err(Error::new(source, "atomic operand type mismatch"));
        }
    }
    Ok(match op {
        AtomicOp::Store => Type::Unit,
        AtomicOp::CompareExchangeWeak => Type::Checked(scalar),
        _ => Type::Scalar(scalar),
    })
}
