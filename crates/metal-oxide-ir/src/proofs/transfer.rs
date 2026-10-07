use super::{
    analysis::Analysis,
    checks,
    domain::{Range, arithmetic, wrap},
    predicate,
    state::{State, Value},
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
        state.assign(&statement.destination, value);
        if let Some(destination) = state.resolve(&statement.destination) {
            if let Some(predicate) = predicate
                && predicate.nodes() <= 64
                && !predicate.uses(&destination)
            {
                state.predicates.insert(destination.clone(), predicate);
            }
            if let Some(alias) = alias
                && !matches!(&alias, predicate::Atom::Place(p) if p.contains(&destination) || destination.contains(p))
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
            Expression::Dim3(fields) => aggregate(state, fields),
            Expression::AggregateUpdate {
                aggregate,
                field,
                value: v,
            } => {
                let mut result = value(aggregate);
                result.0.retain(|p, _| p.first() != Some(field));
                result.field(*field, value(v));
                result
            }
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
            _ => Value::default(),
        })
    }
}

fn aggregate(state: &State<'_>, fields: &[Operand]) -> Value {
    let mut result = Value::default();
    for (index, operand) in fields.iter().enumerate() {
        result.field(index as u32, state.operand_value(operand));
    }
    result
}
