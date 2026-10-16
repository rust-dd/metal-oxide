use super::super::{Result, function::FunctionImporter};
use metal_oxide_ir as ir;
use rustc_span::Span;

impl FunctionImporter<'_, '_> {
    pub(super) fn vector_reduce(
        &mut self,
        name: &str,
        lanes: &[ir::Operand],
        scalar: ir::Scalar,
        initial: Option<&ir::Operand>,
        statements: &mut Vec<ir::Statement>,
        span: Span,
    ) -> Result<ir::Expression> {
        let boolean = matches!(name, "simd_reduce_any" | "simd_reduce_all");
        let result = ir::Type::Scalar(if boolean { ir::Scalar::Bool } else { scalar });
        let mut values = lanes.to_vec();
        if boolean {
            values = values
                .into_iter()
                .map(|value| self.vector_test(value, scalar, statements, span))
                .collect();
        }
        let (mut accumulated, rest) = if let Some(initial) = initial {
            (initial.clone(), values.as_slice())
        } else {
            (values[0].clone(), &values[1..])
        };
        for value in rest {
            let arguments = vec![accumulated, value.clone()];
            let expression = match name {
                "simd_reduce_min" | "simd_reduce_max" if scalar.is_float() => {
                    ir::Expression::Math {
                        op: if name == "simd_reduce_min" {
                            ir::MathOp::Min
                        } else {
                            ir::MathOp::Max
                        },
                        arguments,
                    }
                }
                "simd_reduce_min" | "simd_reduce_max" => ir::Expression::Integer {
                    op: if name == "simd_reduce_min" {
                        ir::IntegerOp::Min
                    } else {
                        ir::IntegerOp::Max
                    },
                    arguments,
                },
                _ => {
                    let op = match name {
                        "simd_reduce_add_ordered" | "simd_reduce_add_unordered" => {
                            ir::BinaryOp::Add
                        }
                        "simd_reduce_mul_ordered" | "simd_reduce_mul_unordered" => {
                            ir::BinaryOp::Mul
                        }
                        "simd_reduce_all" | "simd_reduce_and" => ir::BinaryOp::BitAnd,
                        "simd_reduce_any" | "simd_reduce_or" => ir::BinaryOp::BitOr,
                        "simd_reduce_xor" => ir::BinaryOp::BitXor,
                        _ => return Err((span, format!("unsupported vector reduction: {name}"))),
                    };
                    ir::Expression::Binary(op, arguments[0].clone(), arguments[1].clone())
                }
            };
            accumulated = self.vector_value(result, expression, statements, span);
        }
        Ok(ir::Expression::Use(accumulated))
    }
}
