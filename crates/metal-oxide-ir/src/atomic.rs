use crate::*;

pub(crate) fn atomic_type(
    module: &Module,
    function: &Function,
    op: AtomicOp,
    order: AtomicOrdering,
    failure: AtomicOrdering,
    operands: &[Operand],
    source: &SourceLocation,
) -> Result<Type, Error> {
    if operands.len() != 2 + op.arity() {
        return Err(Error::new(source, "invalid atomic argument count"));
    }
    let ty = |value| operand_type(module, function, value, source);
    let Type::Buffer {
        element: Element::Scalar(scalar),
        access: Access::Atomic,
        address_space,
    } = ty(&operands[0])?
    else {
        return Err(Error::new(
            source,
            "atomic operations require an atomic scalar buffer",
        ));
    };
    op.validate_order(order, failure)
        .map_err(|error| Error::new(source, error))?;
    if scalar
        .abi()
        .and_then(|scalar| op.capability(scalar, address_space == AddressSpace::Threadgroup))
        .is_none()
    {
        return Err(Error::new(
            source,
            "unsupported atomic operation for this element and address space",
        ));
    }
    if ty(&operands[1])? != Type::Scalar(Scalar::U32) {
        return Err(Error::new(source, "atomic index requires u32"));
    }
    for value in &operands[2..] {
        if ty(value)? != Type::Scalar(scalar) {
            return Err(Error::new(source, "atomic operand type mismatch"));
        }
    }
    Ok(match op {
        AtomicOp::Store | AtomicOp::MinStore | AtomicOp::MaxStore => Type::Unit,
        AtomicOp::CompareExchangeWeak => Type::Checked(scalar),
        _ => Type::Scalar(scalar),
    })
}
