mod division;
mod lanes;
mod operations;
mod reductions;
mod validation;

use super::{Result, function::FunctionImporter, location};
use crate::intrinsics::VectorOp;
use lanes::integer;
use metal_oxide_ir as ir;
use rustc_middle::mir;
use rustc_span::{Span, Spanned};

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(in crate::import) fn vector_call(
        &mut self,
        op: VectorOp,
        args: &[Spanned<mir::Operand<'tcx>>],
        destination: (&mir::Place<'tcx>, usize),
        body: &mir::Body<'tcx>,
        statements: &mut Vec<ir::Statement>,
        span: Span,
    ) -> Result<ir::Terminator> {
        let (destination, target) = destination;
        if let VectorOp::Intrinsic(name) = op {
            let types = args
                .iter()
                .map(|arg| self.normalize_type(arg.node.ty(&body.local_decls, self.module.tcx)))
                .collect::<Vec<_>>();
            let result = self.normalize_type(destination.ty(&body.local_decls, self.module.tcx).ty);
            validation::intrinsic(self.module.tcx, name.as_str(), &types, result, span)?;
        }
        let (destination, result) = self.typed_place(destination, span)?;
        let types = args
            .iter()
            .map(|arg| self.operand_type(&arg.node, arg.span))
            .collect::<Result<Vec<_>>>()?;
        let values = args
            .iter()
            .map(|arg| self.operand(&arg.node, arg.span))
            .collect::<Result<Vec<_>>>()?;
        let checks = if let VectorOp::Intrinsic(name) = op
            && matches!(name.as_str(), "simd_div" | "simd_rem")
            && self.vector_shape(types[0], span)?.0.is_integer()
        {
            self.vector_division_checks(name.as_str(), &values, types[0], statements, span)?
        } else {
            Vec::new()
        };
        let start = statements.len();
        let value = match op {
            VectorOp::AsArray(mutable) => {
                let ir::Operand::Place(mut place) = values[0].clone() else {
                    return Err((span, "vector array references require a place".into()));
                };
                place
                    .projection
                    .extend([ir::Projection::Deref, ir::Projection::Field(0)]);
                ir::Expression::Ref {
                    place: ir::Operand::Place(place),
                    mutable,
                }
            }
            VectorOp::FromArray | VectorOp::ToArray => {
                let lanes = self.vector_lanes(values[0].clone(), types[0], span)?;
                ir::Expression::Use(self.vector_pack(result, lanes, statements, span)?)
            }
            VectorOp::MaskFromArray | VectorOp::MaskToArray => {
                let lanes = self.vector_lanes(values[0].clone(), types[0], span)?;
                let (input, _) = self.vector_shape(types[0], span)?;
                let (output, _) = self.vector_shape(result, span)?;
                let lanes = lanes
                    .into_iter()
                    .map(|value| {
                        let test = self.vector_test(value, input, statements, span);
                        self.vector_mask(test, output, statements, span)
                    })
                    .collect();
                ir::Expression::Use(self.vector_pack(result, lanes, statements, span)?)
            }
            VectorOp::ToBitmask => ir::Expression::Use(self.vector_bitmask(
                values[0].clone(),
                types[0],
                ir::Scalar::U64,
                statements,
                span,
            )?),
            VectorOp::FromBitmask => {
                let (scalar, count) = self.vector_shape(result, span)?;
                let mut lanes = Vec::new();
                for lane in 0..count {
                    let bit = self.vector_value(
                        ir::Type::Scalar(ir::Scalar::U64),
                        ir::Expression::Binary(
                            ir::BinaryOp::BitAnd,
                            values[0].clone(),
                            integer(ir::Scalar::U64, 1 << lane),
                        ),
                        statements,
                        span,
                    );
                    let test = self.vector_test(bit, ir::Scalar::U64, statements, span);
                    lanes.push(self.vector_mask(test, scalar, statements, span));
                }
                ir::Expression::Use(self.vector_pack(result, lanes, statements, span)?)
            }
            VectorOp::Intrinsic(name) => {
                self.vector_intrinsic(name.as_str(), &values, &types, result, statements, span)?
            }
        };
        statements.push(ir::Statement {
            destination,
            value,
            source: location(self.module.tcx, span),
        });
        if checks.is_empty() {
            return Ok(ir::Terminator::Goto(target));
        }
        let source = self.check_location(span);
        let mut next = self.append(
            body,
            ir::Block {
                statements: statements.split_off(start),
                terminator: ir::Terminator::Goto(target),
                source: source.clone(),
            },
        );
        let (first, rest) = checks.split_first().unwrap();
        for (condition, check) in rest.iter().rev() {
            next = self.append(
                body,
                ir::Block {
                    statements: Vec::new(),
                    terminator: ir::Terminator::Assert {
                        condition: condition.clone(),
                        expected: false,
                        enabled: true,
                        target: next,
                        check: *check,
                    },
                    source: source.clone(),
                },
            );
        }
        Ok(ir::Terminator::Assert {
            condition: first.0.clone(),
            expected: false,
            enabled: true,
            target: next,
            check: first.1,
        })
    }

    pub(in crate::import) fn vector_array_reference(
        &mut self,
        reference: ir::Type,
        mutable: bool,
        span: Span,
    ) -> Result<ir::Type> {
        let table = self.module.types.table();
        let pointer = table
            .pointer(reference)
            .ok_or((span, "vector array references require a pointer".into()))?;
        let target = table
            .field(pointer.target, 0)
            .ok_or((span, "unsupported vector representation".into()))?;
        Ok(self.module.types.intern_pointer(ir::Pointer {
            target,
            mutable,
            ..pointer
        }))
    }

    fn vector_bitmask(
        &mut self,
        value: ir::Operand,
        ty: ir::Type,
        output: ir::Scalar,
        statements: &mut Vec<ir::Statement>,
        span: Span,
    ) -> Result<ir::Operand> {
        let lanes = self.vector_lanes(value, ty, span)?;
        let (input, _) = self.vector_shape(ty, span)?;
        let mut bits = integer(output, 0);
        for (index, lane) in lanes.into_iter().enumerate() {
            let test = self.vector_test(lane, input, statements, span);
            let bit = self.vector_value(
                ir::Type::Scalar(output),
                ir::Expression::Select {
                    condition: test,
                    yes: integer(output, 1 << index),
                    no: integer(output, 0),
                },
                statements,
                span,
            );
            bits = self.vector_value(
                ir::Type::Scalar(output),
                ir::Expression::Binary(ir::BinaryOp::BitOr, bits, bit),
                statements,
                span,
            );
        }
        Ok(bits)
    }
}
