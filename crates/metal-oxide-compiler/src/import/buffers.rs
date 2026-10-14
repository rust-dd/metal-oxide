use metal_oxide_ir as ir;
use rustc_middle::mir;
use rustc_span::{Span, Spanned};

use super::{Result, function::FunctionImporter, location};
use crate::intrinsics::{BufferQuery, Intrinsic};

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(super) fn buffer_query(
        &self,
        func: &mir::Operand<'tcx>,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<Option<BufferQuery>> {
        Ok(match self.callee(func, body, span)?.intrinsic {
            Some(Intrinsic::Buffer(query)) => Some(query),
            _ => None,
        })
    }

    /// Expands `len`, `is_empty`, `as_slice` or `get_mut` on a kernel buffer parameter, whose
    /// length the launch binds. Returns the terminator of the current block.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn buffer(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        query: BufferQuery,
        args: &[Spanned<mir::Operand<'tcx>>],
        destination: &mir::Place<'tcx>,
        target: usize,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Terminator> {
        let Some(receiver) = args.first() else {
            return Err((span, "invalid buffer query arguments".into()));
        };
        let buffer = ir::Operand::local(self.buffer_parameter(&receiver.node, span)?);
        let (destination, destination_type) = self.typed_place(destination, span)?;
        let source = location(self.module.tcx, span);
        let statement = |destination, value| ir::Statement {
            destination,
            value,
            source: source.clone(),
        };
        match query {
            BufferQuery::Length => {
                statements.push(statement(destination, ir::Expression::Len(buffer)))
            }
            BufferQuery::Slice => {
                statements.push(statement(destination, ir::Expression::Slice(buffer)));
            }
            BufferQuery::IsEmpty => {
                let length = self.temporary(ir::Type::Scalar(ir::Scalar::Usize));
                statements.push(statement(
                    ir::Place::local(length),
                    ir::Expression::Len(buffer),
                ));
                statements.push(statement(
                    destination,
                    ir::Expression::Binary(
                        ir::BinaryOp::Eq,
                        ir::Operand::local(length),
                        ir::Operand::Constant(ir::Constant::Usize(0)),
                    ),
                ));
            }
            BufferQuery::Element => {
                if args.len() != 2 {
                    return Err((span, "invalid buffer element arguments".into()));
                }
                return self.element(
                    statements,
                    buffer,
                    (destination, destination_type),
                    target,
                    body,
                    &source,
                );
            }
        }
        Ok(ir::Terminator::Goto(target))
    }

    /// `Some(&mut buffer[index])` when the thread's index, which its witness holds, is below
    /// the buffer length, else `None`.
    fn element(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        buffer: ir::Operand,
        (option, option_type): (ir::Place, ir::Type),
        target: usize,
        body: &mir::Body<'tcx>,
        source: &ir::SourceLocation,
    ) -> Result<ir::Terminator> {
        let span = rustc_span::DUMMY_SP;
        let table = self.module.types.table();
        let (ir::Type::Aggregate(option_id), Some(pointer)) =
            (option_type, self.module.types.option_value(option_type))
        else {
            return Err((span, "invalid buffer element result".into()));
        };
        let element = table
            .pointer(pointer)
            .ok_or((span, "invalid buffer element result".into()))?;
        let slice = self.module.types.intern_pointer(ir::Pointer {
            slice: true,
            ..element
        });
        let statement = |destination, value| ir::Statement {
            destination,
            value,
            source: source.clone(),
        };
        let position = self.temporary(ir::Type::Scalar(ir::Scalar::Usize));
        let view = self.temporary(slice);
        let length = self.temporary(ir::Type::Scalar(ir::Scalar::Usize));
        let inside = self.temporary(ir::Type::Scalar(ir::Scalar::Bool));
        statements.extend([
            statement(ir::Place::local(position), ir::Expression::ThreadIndex),
            statement(ir::Place::local(view), ir::Expression::Slice(buffer)),
            statement(
                ir::Place::local(length),
                ir::Expression::Len(ir::Operand::local(view)),
            ),
            statement(
                ir::Place::local(inside),
                ir::Expression::Binary(
                    ir::BinaryOp::Lt,
                    ir::Operand::local(position),
                    ir::Operand::local(length),
                ),
            ),
        ]);
        let reference = self.temporary(pointer);
        let some = vec![
            statement(
                ir::Place::local(reference),
                ir::Expression::Ref {
                    place: ir::Operand::Place(ir::Place {
                        local: view,
                        projection: vec![ir::Projection::Deref, ir::Projection::Index(position)],
                    }),
                    mutable: true,
                },
            ),
            statement(
                option.clone(),
                ir::Expression::Variant {
                    ty: option_id,
                    variant: 1,
                    fields: vec![ir::Operand::local(reference)],
                },
            ),
        ];
        let none = vec![statement(
            option,
            ir::Expression::Variant {
                ty: option_id,
                variant: 0,
                fields: Vec::new(),
            },
        )];
        let mut branch = |statements| {
            self.append(
                body,
                ir::Block {
                    statements,
                    terminator: ir::Terminator::Goto(target),
                    source: source.clone(),
                },
            )
        };
        let (then_block, else_block) = (branch(some), branch(none));
        Ok(ir::Terminator::Branch {
            condition: ir::Operand::local(inside),
            then_block,
            else_block,
        })
    }

    /// The kernel buffer parameter that a `&buffer` receiver borrows.
    pub(super) fn buffer_parameter(
        &self,
        receiver: &mir::Operand<'tcx>,
        span: Span,
    ) -> Result<usize> {
        let parameter = receiver
            .place()
            .and_then(|place| place.as_local())
            .and_then(|local| self.handles[local.as_usize()])
            .map(|local| local.as_usize())
            .filter(|&local| self.kernel && (1..=self.parameter_count()).contains(&local));
        parameter.ok_or((
            span,
            "buffer lengths are available on kernel buffer parameters in the kernel entrypoint"
                .into(),
        ))
    }

    /// The slice type of a buffer parameter: shared for a read buffer, mutable otherwise.
    pub(super) fn buffer_slice_type(&mut self, parameter: usize, span: Span) -> Result<ir::Type> {
        let ir::Type::Buffer {
            element, access, ..
        } = self.locals[parameter]
        else {
            return Err((span, "buffer slices require a buffer".into()));
        };
        Ok(self.module.types.intern_pointer(ir::Pointer {
            target: element.ty(),
            space: ir::AddressSpace::Device,
            mutable: access != ir::Access::Read,
            slice: true,
        }))
    }
}
