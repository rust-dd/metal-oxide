use super::super::{Result, function::FunctionImporter};
use super::lanes::integer;
use metal_oxide_ir as ir;
use rustc_span::Span;

impl FunctionImporter<'_, '_> {
    pub(super) fn vector_intrinsic(
        &mut self,
        name: &str,
        values: &[ir::Operand],
        types: &[ir::Type],
        result: ir::Type,
        statements: &mut Vec<ir::Statement>,
        span: Span,
    ) -> Result<ir::Expression> {
        if name == "simd_splat" {
            let (_, count) = self.vector_shape(result, span)?;
            return Ok(ir::Expression::Use(self.vector_pack(
                result,
                vec![values[0].clone(); count as usize],
                statements,
                span,
            )?));
        }
        let first = self.vector_lanes(values[0].clone(), types[0], span)?;
        let (input, _) = self.vector_shape(types[0], span)?;
        if name.starts_with("simd_reduce_") {
            return self.vector_reduce(name, &first, input, values.get(1), statements, span);
        }
        if name == "simd_bitmask" {
            let ir::Type::Scalar(output) = result else {
                return Err((span, "vector bitmask requires an integer result".into()));
            };
            return Ok(ir::Expression::Use(self.vector_bitmask(
                values[0].clone(),
                types[0],
                output,
                statements,
                span,
            )?));
        }
        if matches!(name, "simd_extract" | "simd_extract_dyn") {
            let value = self.vector_extract(&first, values[1].clone(), input, statements, span);
            return Ok(ir::Expression::Use(value));
        }
        let (output, count) = self.vector_shape(result, span)?;
        if matches!(name, "simd_insert" | "simd_insert_dyn") {
            let mut lanes = Vec::new();
            for (index, old) in first.into_iter().enumerate() {
                let condition = self.vector_value(
                    ir::Type::Scalar(ir::Scalar::Bool),
                    ir::Expression::Binary(
                        ir::BinaryOp::Eq,
                        values[1].clone(),
                        integer(ir::Scalar::U32, index as u128),
                    ),
                    statements,
                    span,
                );
                lanes.push(self.vector_value(
                    ir::Type::Scalar(output),
                    ir::Expression::Select {
                        condition,
                        yes: values[2].clone(),
                        no: old,
                    },
                    statements,
                    span,
                ));
            }
            return Ok(ir::Expression::Use(
                self.vector_pack(result, lanes, statements, span)?,
            ));
        }
        if name == "simd_shuffle" {
            let second = self.vector_lanes(values[1].clone(), types[1], span)?;
            let indices = self.vector_lanes(values[2].clone(), types[2], span)?;
            let input = first.into_iter().chain(second).collect::<Vec<_>>();
            let lanes = indices
                .iter()
                .map(|index| match index {
                    ir::Operand::Constant(ir::Constant::U32(index)) => input
                        .get(*index as usize)
                        .cloned()
                        .ok_or((span, "vector shuffle index is out of bounds".into())),
                    _ => Err((span, "vector shuffle requires constant indices".into())),
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(ir::Expression::Use(
                self.vector_pack(result, lanes, statements, span)?,
            ));
        }
        let operands = values
            .iter()
            .zip(types)
            .map(|(value, ty)| self.vector_lanes(value.clone(), *ty, span))
            .collect::<Result<Vec<_>>>()?;
        if operands.iter().any(|lanes| lanes.len() != count as usize) {
            return Err((
                span,
                "vector operands must have matching lane counts".into(),
            ));
        }
        let mut lanes = Vec::new();
        for lane in 0..count as usize {
            let args = operands
                .iter()
                .map(|values| values[lane].clone())
                .collect::<Vec<_>>();
            let value = if let Some(op) = binary(name) {
                let comparison = matches!(
                    op,
                    ir::BinaryOp::Eq
                        | ir::BinaryOp::Ne
                        | ir::BinaryOp::Lt
                        | ir::BinaryOp::Le
                        | ir::BinaryOp::Gt
                        | ir::BinaryOp::Ge
                );
                if comparison {
                    let condition = self.vector_value(
                        ir::Type::Scalar(ir::Scalar::Bool),
                        ir::Expression::Binary(op, args[0].clone(), args[1].clone()),
                        statements,
                        span,
                    );
                    lanes.push(self.vector_mask(condition, output, statements, span));
                    continue;
                }
                ir::Expression::Binary(op, args[0].clone(), args[1].clone())
            } else if let Some(op) = integer_op(name) {
                let count = matches!(
                    op,
                    ir::IntegerOp::CountOnes
                        | ir::IntegerOp::LeadingZeros
                        | ir::IntegerOp::TrailingZeros
                );
                let value = ir::Expression::Integer {
                    op,
                    arguments: args,
                };
                if count {
                    let value = self.vector_value(
                        ir::Type::Scalar(ir::Scalar::U32),
                        value,
                        statements,
                        span,
                    );
                    ir::Expression::Cast(value, output)
                } else {
                    value
                }
            } else {
                match name {
                    "simd_neg" => ir::Expression::Unary(ir::UnaryOp::Neg, args[0].clone()),
                    "simd_fabs" => ir::Expression::Math {
                        op: ir::MathOp::Abs,
                        arguments: args,
                    },
                    "simd_minimum_number_nsz" => ir::Expression::Math {
                        op: ir::MathOp::Min,
                        arguments: args,
                    },
                    "simd_maximum_number_nsz" => ir::Expression::Math {
                        op: ir::MathOp::Max,
                        arguments: args,
                    },
                    "simd_as" | "simd_cast" => {
                        if name == "simd_cast" && input.is_float() && output.is_integer() {
                            ir::Expression::UncheckedCast(args[0].clone(), output)
                        } else {
                            ir::Expression::Cast(args[0].clone(), output)
                        }
                    }
                    "simd_select" => {
                        let condition = self.vector_test(args[0].clone(), input, statements, span);
                        ir::Expression::Select {
                            condition,
                            yes: args[1].clone(),
                            no: args[2].clone(),
                        }
                    }
                    _ => return Err((span, format!("unsupported vector intrinsic: {name}"))),
                }
            };
            lanes.push(self.vector_value(ir::Type::Scalar(output), value, statements, span));
        }
        Ok(ir::Expression::Use(
            self.vector_pack(result, lanes, statements, span)?,
        ))
    }

    fn vector_extract(
        &mut self,
        lanes: &[ir::Operand],
        index: ir::Operand,
        scalar: ir::Scalar,
        statements: &mut Vec<ir::Statement>,
        span: Span,
    ) -> ir::Operand {
        let mut result = lanes[lanes.len() - 1].clone();
        for (lane, value) in lanes[..lanes.len() - 1].iter().enumerate().rev() {
            let condition = self.vector_value(
                ir::Type::Scalar(ir::Scalar::Bool),
                ir::Expression::Binary(
                    ir::BinaryOp::Eq,
                    index.clone(),
                    integer(ir::Scalar::U32, lane as u128),
                ),
                statements,
                span,
            );
            result = self.vector_value(
                ir::Type::Scalar(scalar),
                ir::Expression::Select {
                    condition,
                    yes: value.clone(),
                    no: result,
                },
                statements,
                span,
            );
        }
        result
    }
}

fn binary(name: &str) -> Option<ir::BinaryOp> {
    use ir::BinaryOp::*;
    Some(match name {
        "simd_add" => Add,
        "simd_sub" => Sub,
        "simd_mul" => Mul,
        "simd_div" => Div,
        "simd_rem" => Rem,
        "simd_and" => BitAnd,
        "simd_or" => BitOr,
        "simd_xor" => BitXor,
        "simd_shl" => Shl,
        "simd_shr" => Shr,
        "simd_eq" => Eq,
        "simd_ne" => Ne,
        "simd_lt" => Lt,
        "simd_le" => Le,
        "simd_gt" => Gt,
        "simd_ge" => Ge,
        _ => return None,
    })
}

fn integer_op(name: &str) -> Option<ir::IntegerOp> {
    use ir::IntegerOp::*;
    Some(match name {
        "simd_saturating_add" => SaturatingAdd,
        "simd_saturating_sub" => SaturatingSub,
        "simd_ctpop" => CountOnes,
        "simd_ctlz" => LeadingZeros,
        "simd_cttz" => TrailingZeros,
        "simd_bitreverse" => ReverseBits,
        _ => return None,
    })
}
