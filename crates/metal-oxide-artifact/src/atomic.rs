use crate::{Capability, Scalar};

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
    MinStore,
    MaxStore,
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
            Self::MinStore => "min",
            Self::MaxStore => "max",
        }
    }
}

/// Ordering of an atomic operation. Sequential consistency is rejected by the compiler.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AtomicOrdering {
    #[default]
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl AtomicOrdering {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Relaxed => "relaxed",
            Self::Acquire => "acquire",
            Self::Release => "release",
            Self::AcqRel => "acq_rel",
            Self::SeqCst => "seq_cst",
        }
    }
}

impl AtomicOp {
    /// The feature required for this operation and address space, or an unsupported combination.
    pub fn capability(self, scalar: Scalar, threadgroup: bool) -> Option<&'static Capability> {
        use AtomicOp::*;
        let feature = match (scalar, threadgroup, self) {
            (
                Scalar::U32 | Scalar::I32,
                _,
                Load | Store | Exchange | CompareExchangeWeak | Add | Sub | Min | Max | And | Or
                | Xor,
            ) => "int32_atomics",
            (Scalar::F32, false, Load | Store | Exchange | CompareExchangeWeak | Add | Sub) => {
                "float32_atomics"
            }
            (Scalar::U64, false, MinStore | MaxStore) => "uint64_atomics",
            _ => return None,
        };
        Capability::named(feature)
    }

    pub fn validate_order(
        self,
        success: AtomicOrdering,
        failure: AtomicOrdering,
    ) -> Result<(), &'static str> {
        use AtomicOrdering::*;
        if success == SeqCst || failure == SeqCst {
            return Err("sequentially consistent atomics are unsupported");
        }
        if (self == Self::Load && matches!(success, Release | AcqRel))
            || (self == Self::Store && matches!(success, Acquire | AcqRel))
            || (matches!(self, Self::MinStore | Self::MaxStore) && success != Relaxed)
            || (self != Self::CompareExchangeWeak && failure != Relaxed)
            || matches!(failure, Release | AcqRel)
            || (failure == Acquire && !matches!(success, Acquire | AcqRel))
        {
            return Err("invalid atomic memory ordering");
        }
        Ok(())
    }
}
