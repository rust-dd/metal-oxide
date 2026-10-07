use super::{
    domain::Range,
    predicate::{Atom, Comparison, Predicate},
    state::State,
};
use crate::*;

pub(super) fn place(
    state: &State<'_>,
    place: &Place,
    source: &SourceLocation,
) -> Result<(), Error> {
    let mut ty = state.function.locals[place.local];
    for projection in &place.projection {
        ty = match *projection {
            Projection::Field(field) => state.module.types.field(ty, field).unwrap(),
            Projection::Index(index) => {
                let Type::Aggregate(id) = ty else {
                    unreachable!()
                };
                let Aggregate::Array { element, length } = state.module.types.get(id).unwrap()
                else {
                    unreachable!()
                };
                let range = state.range(&Place::local(index)).unwrap();
                if range.lo < 0 || range.hi >= i128::from(*length) {
                    return Err(Error::new(
                        source,
                        "dynamic array index bounds cannot be proved safe",
                    ));
                }
                *element
            }
        };
    }
    Ok(())
}

pub(super) fn division(
    state: &State<'_>,
    a: &Operand,
    b: &Operand,
    source: &SourceLocation,
) -> Result<(), Error> {
    let Type::Scalar(scalar) = operand_type(state.module, state.function, a, source)? else {
        unreachable!()
    };
    if !scalar.is_integer() {
        return Ok(());
    }
    let left = state.operand_range(a).unwrap();
    let right = state.operand_range(b).unwrap();
    if right.contains(0) {
        return Err(Error::new(
            source,
            "integer division/remainder requires a proved nonzero divisor",
        ));
    }
    let minimum = Range::full(scalar).lo;
    if scalar.is_signed() && left.contains(minimum) && right.contains(-1) {
        let eq = |operand, value| {
            let mut a = state.symbol(operand)?;
            let mut b = Atom::Integer(value);
            if b < a {
                std::mem::swap(&mut a, &mut b);
            }
            Some(Predicate::Compare(Comparison::Eq, a, b))
        };
        let excluded = eq(a, minimum)
            .zip(eq(b, -1))
            .is_some_and(|(a, b)| Predicate::combine(a, b, true).value(state).hi == 0);
        if !excluded {
            return Err(Error::new(
                source,
                "signed division/remainder requires a proved MIN/-1 exclusion",
            ));
        }
    }
    Ok(())
}
