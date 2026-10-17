use crate::*;

/// An operation on 8x8 matrices that the lanes of a SIMD group hold together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixOp {
    /// `[value]`: every element set to `value`.
    Splat,
    /// `[a, b, c]`: `a * b + c` in the element type of `c`.
    MultiplyAccumulate,
    /// `[buffer, row, column, stride]`: the block at `(row, column)` of a row-major matrix.
    Load { transposed: bool },
    /// `[buffer, row, column, stride, matrix]`.
    Store { transposed: bool },
}

impl MatrixOp {
    pub const fn arity(self) -> usize {
        match self {
            Self::Splat => 1,
            Self::MultiplyAccumulate => 3,
            Self::Load { .. } => 4,
            Self::Store { .. } => 5,
        }
    }
}

/// The element types of SIMD-group matrices.
pub const fn is_matrix_element(scalar: Scalar) -> bool {
    matches!(scalar, Scalar::F32 | Scalar::F16 | Scalar::BF16)
}

pub(crate) fn matrix_type(
    module: &Module,
    function: &Function,
    op: MatrixOp,
    operands: &[Operand],
    source: &SourceLocation,
) -> Result<Type, Error> {
    let types = operands
        .iter()
        .map(|operand| operand_type(module, function, operand, source))
        .collect::<Result<Vec<_>, _>>()?;
    result(op, &types).ok_or_else(|| Error::new(source, "invalid SIMD-group matrix operation"))
}

fn result(op: MatrixOp, types: &[Type]) -> Option<Type> {
    let element = |ty: &Type| match *ty {
        Type::Matrix(scalar) => Some(scalar),
        _ => None,
    };
    let positions =
        |positions: [&Type; 3]| positions.iter().all(|&&ty| ty == Type::Scalar(Scalar::U32));
    match (op, types) {
        (MatrixOp::Splat, [Type::Scalar(scalar)]) => {
            is_matrix_element(*scalar).then_some(Type::Matrix(*scalar))
        }
        (MatrixOp::MultiplyAccumulate, [a, b, c]) => {
            element(a)?;
            element(b)?;
            Some(Type::Matrix(element(c)?))
        }
        (MatrixOp::Load { .. }, [buffer, row, column, stride]) => {
            let scalar = memory(buffer, false)?;
            positions([row, column, stride]).then_some(Type::Matrix(scalar))
        }
        (MatrixOp::Store { .. }, [buffer, row, column, stride, value]) => {
            let scalar = memory(buffer, true)?;
            (positions([row, column, stride]) && element(value) == Some(scalar))
                .then_some(Type::Unit)
        }
        _ => None,
    }
}

/// The element of a buffer that matrices may be loaded from, or stored to when `store` is set.
fn memory(buffer: &Type, store: bool) -> Option<Scalar> {
    let Type::Buffer {
        element: Element::Scalar(scalar),
        access,
        address_space,
    } = *buffer
    else {
        return None;
    };
    let accessible = match access {
        Access::ReadWrite => address_space != AddressSpace::Thread,
        Access::Read => !store && address_space == AddressSpace::Device,
        Access::Write => store && address_space == AddressSpace::Device,
        Access::Atomic => false,
    };
    (accessible && is_matrix_element(scalar)).then_some(scalar)
}
