use super::super::{Result, function::FunctionImporter};
use super::lanes::integer;
use metal_oxide_ir as ir;
use rustc_span::Span;

impl FunctionImporter<'_, '_> {
    /// Scalar proofs do not retain correlations across vector masks and helper returns.
    pub(super) fn vector_division_checks(
        &mut self,
        name: &str,
        values: &[ir::Operand],
        ty: ir::Type,
        statements: &mut Vec<ir::Statement>,
        span: Span,
    ) -> Result<Vec<(ir::Operand, ir::Check)>> {
        let (scalar, _) = self.vector_shape(ty, span)?;
        let left = self.vector_lanes(values[0].clone(), ty, span)?;
        let right = self.vector_lanes(values[1].clone(), ty, span)?;
        let remainder = name == "simd_rem";
        let mut checks = Vec::new();
        for (left, right) in left.into_iter().zip(right) {
            let zero = self.vector_value(
                ir::Type::Scalar(ir::Scalar::Bool),
                ir::Expression::Binary(ir::BinaryOp::Eq, right.clone(), integer(scalar, 0)),
                statements,
                span,
            );
            checks.push((
                zero,
                if remainder {
                    ir::Check::RemainderByZero
                } else {
                    ir::Check::DivisionByZero
                },
            ));
            if scalar.is_signed() {
                let minimum = self.vector_value(
                    ir::Type::Scalar(ir::Scalar::Bool),
                    ir::Expression::Binary(
                        ir::BinaryOp::Eq,
                        left,
                        integer(scalar, 1 << (scalar.bits() - 1)),
                    ),
                    statements,
                    span,
                );
                let minus_one = self.vector_value(
                    ir::Type::Scalar(ir::Scalar::Bool),
                    ir::Expression::Binary(ir::BinaryOp::Eq, right, integer(scalar, u128::MAX)),
                    statements,
                    span,
                );
                let overflow = self.vector_value(
                    ir::Type::Scalar(ir::Scalar::Bool),
                    ir::Expression::Binary(ir::BinaryOp::BitAnd, minimum, minus_one),
                    statements,
                    span,
                );
                checks.push((
                    overflow,
                    ir::Check::Overflow(if remainder {
                        ir::OverflowOp::Rem
                    } else {
                        ir::OverflowOp::Div
                    }),
                ));
            }
        }
        Ok(checks)
    }
}
