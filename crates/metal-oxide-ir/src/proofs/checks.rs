use super::{
    domain::Range,
    predicate::{Atom, Comparison, Predicate},
    state::State,
};
use crate::{typing::Located, *};

pub(super) fn place(
    state: &State<'_>,
    place: &Place,
    source: &SourceLocation,
) -> Result<(), Error> {
    let types = &state.module.types;
    let mut located = Located::Value(state.function.locals[place.local]);
    for (position, projection) in place.projection.iter().enumerate() {
        located = match (*projection, located) {
            (Projection::Field(field), Located::Value(ty)) => {
                Located::Value(types.field(ty, field).unwrap())
            }
            (Projection::Index(index), Located::Value(Type::Aggregate(id))) => {
                let Aggregate::Array { element, length } = types.get(id).unwrap() else {
                    unreachable!()
                };
                let range = state.range(&Place::local(index)).unwrap();
                if range.lo < 0 || range.hi >= i128::from(*length) {
                    return Err(Error::new(
                        source,
                        "dynamic array index bounds cannot be proved safe",
                    ));
                }
                Located::Value(*element)
            }
            (Projection::Index(index), Located::Elements(slice)) => {
                // The projection before the index is the slice's dereference.
                let slice_place = Place {
                    local: place.local,
                    projection: place.projection[..position - 1].to_vec(),
                };
                if !state
                    .index_atoms(index)
                    .iter()
                    .any(|atom| state.bounded(atom, Comparison::Lt, &slice_place))
                {
                    return Err(Error::new(
                        source,
                        "slice index is not below a checked slice length",
                    ));
                }
                Located::Value(slice.target)
            }
            (Projection::Deref, Located::Value(ty)) => {
                let pointer = types.pointer(ty).unwrap();
                if pointer.slice {
                    Located::Elements(pointer)
                } else {
                    Located::Value(pointer.target)
                }
            }
            _ => unreachable!("validated place"),
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
    let nonzero =
        !right.contains(0) || equality(state, b, 0).is_some_and(|zero| zero.value(state).hi == 0);
    if !nonzero {
        return Err(Error::new(
            source,
            "integer division/remainder requires a proved nonzero divisor",
        ));
    }
    let minimum = Range::full(scalar).lo;
    if scalar.is_signed() && left.contains(minimum) && right.contains(-1) {
        let excluded = equality(state, a, minimum)
            .zip(equality(state, b, -1))
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

/// `operand == value` in the canonical atom order that assumed conditions are recorded under.
fn equality(state: &State<'_>, operand: &Operand, value: i128) -> Option<Predicate> {
    let mut a = state.symbol(operand)?;
    let mut b = Atom::Integer(value);
    if b < a {
        std::mem::swap(&mut a, &mut b);
    }
    Some(Predicate::Compare(Comparison::Eq, a, b))
}

/// `slice[start..end]` needs its range checks: `start <= end <= len`.
pub(super) fn subslice(
    state: &State<'_>,
    slice: &Operand,
    start: &Operand,
    end: &Operand,
    source: &SourceLocation,
) -> Result<(), Error> {
    let (Operand::Place(slice), Some(start), Some(end)) =
        (slice, state.written(start), state.written(end))
    else {
        return Err(Error::new(source, "subslice bounds are not checked"));
    };
    if !state.ordered(&start, &end) || !state.bounded(&end, Comparison::Le, slice) {
        return Err(Error::new(source, "subslice bounds are not checked"));
    }
    Ok(())
}
