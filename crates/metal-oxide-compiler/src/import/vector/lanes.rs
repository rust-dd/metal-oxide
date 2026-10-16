use super::super::{Result, function::FunctionImporter, location};
use metal_oxide_ir as ir;
use rustc_span::Span;

impl FunctionImporter<'_, '_> {
    pub(super) fn vector_lanes(
        &self,
        value: ir::Operand,
        ty: ir::Type,
        span: Span,
    ) -> Result<Vec<ir::Operand>> {
        match ty {
            ir::Type::Aggregate(id) => match self.module.types.table().get(id).unwrap() {
                ir::Aggregate::Array { length, .. } => (0..*length)
                    .map(|index| field(&value, index, span))
                    .collect(),
                ir::Aggregate::Record { fields, .. } if fields.len() == 1 => {
                    self.vector_lanes(field(&value, 0, span)?, fields[0].ty, span)
                }
                _ => Err((span, "unsupported vector representation".into())),
            },
            _ => Err((span, "vector operation requires a vector".into())),
        }
    }

    pub(super) fn vector_shape(&self, ty: ir::Type, span: Span) -> Result<(ir::Scalar, u32)> {
        let ir::Type::Aggregate(id) = ty else {
            return Err((span, "vector operation requires a vector".into()));
        };
        match self.module.types.table().get(id).unwrap() {
            ir::Aggregate::Array {
                element: ir::Type::Scalar(scalar),
                length,
            } => Ok((*scalar, *length)),
            ir::Aggregate::Record { fields, .. } if fields.len() == 1 => {
                self.vector_shape(fields[0].ty, span)
            }
            _ => Err((span, "unsupported vector representation".into())),
        }
    }

    pub(super) fn vector_pack(
        &mut self,
        ty: ir::Type,
        lanes: Vec<ir::Operand>,
        statements: &mut Vec<ir::Statement>,
        span: Span,
    ) -> Result<ir::Operand> {
        let ir::Type::Aggregate(id) = ty else {
            return Err((span, "vector operation requires a vector".into()));
        };
        let fields = match self.module.types.table().get(id).unwrap().clone() {
            ir::Aggregate::Array { length, .. } if length as usize == lanes.len() => lanes,
            ir::Aggregate::Record { fields, .. } if fields.len() == 1 => {
                vec![self.vector_pack(fields[0].ty, lanes, statements, span)?]
            }
            _ => {
                return Err((
                    span,
                    "vector lane count does not match the destination".into(),
                ));
            }
        };
        Ok(self.vector_value(
            ty,
            ir::Expression::Aggregate { ty: id, fields },
            statements,
            span,
        ))
    }

    pub(super) fn vector_value(
        &mut self,
        ty: ir::Type,
        value: ir::Expression,
        statements: &mut Vec<ir::Statement>,
        span: Span,
    ) -> ir::Operand {
        let local = self.temporary(ty);
        let place = ir::Place::local(local);
        statements.push(ir::Statement {
            destination: place.clone(),
            value,
            source: location(self.module.tcx, span),
        });
        ir::Operand::Place(place)
    }

    pub(super) fn vector_test(
        &mut self,
        value: ir::Operand,
        scalar: ir::Scalar,
        statements: &mut Vec<ir::Statement>,
        span: Span,
    ) -> ir::Operand {
        if scalar == ir::Scalar::Bool {
            return value;
        }
        self.vector_value(
            ir::Type::Scalar(ir::Scalar::Bool),
            ir::Expression::Binary(ir::BinaryOp::Ne, value, integer(scalar, 0)),
            statements,
            span,
        )
    }

    pub(super) fn vector_mask(
        &mut self,
        test: ir::Operand,
        scalar: ir::Scalar,
        statements: &mut Vec<ir::Statement>,
        span: Span,
    ) -> ir::Operand {
        if scalar == ir::Scalar::Bool {
            return test;
        }
        self.vector_value(
            ir::Type::Scalar(scalar),
            ir::Expression::Select {
                condition: test,
                yes: integer(scalar, u128::MAX),
                no: integer(scalar, 0),
            },
            statements,
            span,
        )
    }
}

pub(super) fn integer(scalar: ir::Scalar, value: u128) -> ir::Operand {
    ir::Operand::Constant(
        super::super::control::integer(scalar, value).expect("integer vector element"),
    )
}

pub(super) fn field(value: &ir::Operand, index: u32, span: Span) -> Result<ir::Operand> {
    match value {
        ir::Operand::Place(place) => {
            let mut place = place.clone();
            place.projection.push(ir::Projection::Field(index));
            Ok(ir::Operand::Place(place))
        }
        ir::Operand::AggregateConstant { fields, .. } => fields
            .get(index as usize)
            .cloned()
            .ok_or((span, "invalid vector field".into())),
        _ => Err((span, "vector operation requires a vector".into())),
    }
}
