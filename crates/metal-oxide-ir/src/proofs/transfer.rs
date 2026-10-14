use super::{
    analysis::Analysis,
    checks,
    domain::{Range, arithmetic, wrap},
    predicate,
    state::{REFERENT, State, Value},
};
use crate::*;

impl Analysis<'_> {
    pub fn statement(&mut self, state: &mut State<'_>, statement: &Statement) -> Result<(), Error> {
        checks::place(state, &statement.destination, &statement.source)?;
        for operand in statement.value.operands() {
            if let Operand::Place(p) = operand {
                checks::place(state, p, &statement.source)?;
            }
        }
        if let Expression::Subslice { slice, start, end } = &statement.value {
            checks::subslice(state, slice, start, end, &statement.source)?;
        }
        let ty = place_type(
            state.module,
            state.function,
            &statement.destination,
            &statement.source,
        )?;
        let value = self.expression(state, &statement.value, ty, &statement.source)?;
        let predicate = if ty == Type::Scalar(Scalar::Bool) {
            match &statement.value {
                Expression::Use(operand) => state.predicate(operand),
                Expression::Unary(UnaryOp::Not, operand) => {
                    state.predicate(operand).map(|p| p.negate())
                }
                Expression::Binary(op, a, b) => predicate::from_binary(state, *op, a, b),
                _ => None,
            }
        } else {
            None
        };
        let alias = match &statement.value {
            Expression::Use(operand) => state.symbol(operand),
            _ => None,
        };
        if matches!(statement.value, Expression::Call { .. }) {
            state.forget_escaped();
        }
        state.assign(&statement.destination, value);
        if statement.value == Expression::ThreadIndex
            && let Some(destination) = state.resolve(&statement.destination)
        {
            state.threads.insert(destination);
        }
        match &statement.value {
            Expression::Len(Operand::Place(slice)) => state.length(&statement.destination, slice),
            Expression::Use(Operand::Place(source)) => {
                state.copy(&statement.destination, source);
                if pointer(state, source) {
                    state.reborrow(&statement.destination, source);
                }
            }
            Expression::Ref {
                place: Operand::Place(place),
                ..
            } if place.projection.last() == Some(&Projection::Deref) => {
                let mut slice = place.clone();
                slice.projection.pop();
                if pointer(state, &slice) {
                    state.reborrow(&statement.destination, &slice);
                }
            }
            Expression::Binary(op, a, b) => {
                if let Some(comparison) = predicate::comparison(*op) {
                    state.record_comparison(&statement.destination, comparison, a, b);
                }
            }
            Expression::Subslice { start, end, .. } => {
                // The bounds check before a subslice proves `start <= end`.
                if let (Some(start), Some(end)) =
                    (state.operand_range(start), state.operand_range(end))
                    && end.hi >= start.lo
                {
                    let length = Range::new((end.lo - start.hi).max(0), end.hi - start.lo);
                    state.extent_range(&statement.destination, length);
                }
            }
            Expression::Aggregate { fields, .. } | Expression::Variant { fields, .. } => {
                for (index, field) in fields.iter().enumerate() {
                    if let Operand::Place(source) = field {
                        let mut destination = statement.destination.clone();
                        destination.projection.push(Projection::Field(component(
                            state,
                            &statement.value,
                            index,
                        )));
                        state.copy(&destination, source);
                        if pointer(state, source) {
                            state.reborrow(&destination, source);
                        }
                    }
                }
            }
            Expression::Slice(data) => {
                let ty = operand_type(state.module, state.function, data, &statement.source).ok();
                if let (Some(Type::Buffer { .. }), Operand::Place(buffer)) = (ty, data) {
                    // A buffer slice has the buffer's length.
                    state.reborrow(&statement.destination, buffer);
                }
                let length = ty
                    .and_then(|ty| state.module.types.pointer(ty))
                    .and_then(|pointer| match pointer.target {
                        Type::Aggregate(id) => state.module.types.get(id).map(Aggregate::len),
                        _ => None,
                    });
                if let Some(length) = length {
                    state.extent(&statement.destination, length as u32);
                }
            }
            _ => {}
        }
        if let Some(destination) = state.resolve(&statement.destination) {
            if let Some(predicate) = predicate
                && predicate.nodes() <= 64
                && !predicate.uses(&destination)
            {
                state.predicates.insert(destination.clone(), predicate);
            }
            if let Some(alias) = alias
                && !alias.uses(&destination)
            {
                state.aliases.insert(destination, alias);
            }
        }
        Ok(())
    }

    fn expression(
        &mut self,
        state: &State<'_>,
        expression: &Expression,
        ty: Type,
        source: &SourceLocation,
    ) -> Result<Value, Error> {
        let range = |operand| state.operand_range(operand);
        let value = |operand| state.operand_value(operand);
        Ok(match expression {
            Expression::Use(operand) => value(operand),
            Expression::Checked {
                value: v, overflow, ..
            } => {
                let mut result = Value::default();
                result.field(0, value(v));
                result.field(1, value(overflow));
                result
            }
            Expression::Aggregate { fields, .. } => aggregate(state, fields),
            Expression::Variant {
                variant, fields, ..
            } => {
                let mut result = Value::default();
                result.field(0, Value::scalar(Range::exact(i128::from(*variant))));
                for (index, operand) in fields.iter().enumerate() {
                    result.field(component(state, expression, index), value(operand));
                }
                result
            }
            Expression::Dim3(fields) => aggregate(state, fields),
            Expression::Ref {
                place,
                mutable: false,
            } => {
                let mut result = Value::default();
                result.field(REFERENT, value(place));
                result
            }
            Expression::Coordinates(builtin) => {
                let mut result = Value::default();
                for axis in 0..3 {
                    let block = self.block.map(|block| block[axis as usize]);
                    result.field(
                        axis,
                        Value::scalar(Range::coordinate(*builtin, axis, block)),
                    );
                }
                result
            }
            // The runtime caps each launch axis at `u32::MAX` threads.
            Expression::ThreadIndex => Value::scalar(Range::new(0, i128::from(u32::MAX) - 1)),
            Expression::Cast(operand, to) if to.is_integer() || *to == Scalar::Bool => {
                if let Some(from) = range(operand) {
                    let full = Range::full(*to);
                    Value::scalar(if from.lo >= full.lo && from.hi <= full.hi {
                        from
                    } else if let Some(n) = from.singleton() {
                        Range::exact(wrap(n, *to))
                    } else {
                        full
                    })
                } else {
                    Value::default()
                }
            }
            Expression::Unary(op, operand) => {
                if let Some(from) = range(operand) {
                    let Type::Scalar(scalar) = ty else {
                        unreachable!()
                    };
                    let full = Range::full(scalar);
                    let result = match op {
                        UnaryOp::Not if scalar == Scalar::Bool => {
                            Range::new(1 - from.hi, 1 - from.lo)
                        }
                        UnaryOp::Neg if -from.hi >= full.lo && -from.lo <= full.hi => {
                            Range::new(-from.hi, -from.lo)
                        }
                        _ => from.singleton().map_or(full, |n| {
                            Range::exact(wrap(if *op == UnaryOp::Neg { -n } else { !n }, scalar))
                        }),
                    };
                    Value::scalar(result)
                } else {
                    Value::default()
                }
            }
            Expression::Binary(op, a, b) => {
                if matches!(op, BinaryOp::Div | BinaryOp::Rem) {
                    checks::division(state, a, b, source)?;
                }
                if let Some((a, b)) = range(a).zip(range(b)) {
                    if let Some(cmp) = predicate::comparison(*op) {
                        Value::scalar(predicate::compare(cmp, &a, &b))
                    } else {
                        let scalar = match ty {
                            Type::Scalar(s) | Type::Checked(s) => s,
                            _ => unreachable!(),
                        };
                        let (value, overflow) = arithmetic(*op, &a, &b, scalar);
                        if matches!(ty, Type::Checked(_)) {
                            let mut result = Value::default();
                            result.field(0, Value::scalar(value));
                            result.field(1, Value::scalar(overflow));
                            result
                        } else {
                            Value::scalar(value)
                        }
                    }
                } else {
                    Value::default()
                }
            }
            Expression::Call {
                function,
                arguments,
            } => self.function(*function, &arguments.iter().map(value).collect::<Vec<_>>())?,
            Expression::Len(Operand::Place(slice)) => state
                .length_range(slice)
                .map_or_else(Value::default, Value::scalar),
            _ => Value::default(),
        })
    }
}

/// The field of the aggregate that `expression` builds which holds its operand `index`.
fn component(state: &State<'_>, expression: &Expression, index: usize) -> u32 {
    match expression {
        Expression::Variant { ty, variant, .. } => state
            .module
            .types
            .get(*ty)
            .and_then(|aggregate| aggregate.payload(*variant, index as u32))
            .expect("typed variants have payload slots"),
        _ => index as u32,
    }
}

fn aggregate(state: &State<'_>, fields: &[Operand]) -> Value {
    let mut result = Value::default();
    for (index, operand) in fields.iter().enumerate() {
        result.field(index as u32, state.operand_value(operand));
    }
    result
}

/// Whether `place` holds a slice.
fn pointer(state: &State<'_>, place: &Place) -> bool {
    place_type(state.module, state.function, place, &state.function.source)
        .ok()
        .and_then(|ty| state.module.types.pointer(ty))
        .is_some_and(|pointer| pointer.slice)
}
