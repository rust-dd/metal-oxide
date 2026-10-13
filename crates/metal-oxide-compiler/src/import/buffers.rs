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

    /// Expands `len`, `is_empty` or `as_slice` on a kernel buffer parameter, whose length the
    /// launch binds.
    pub(super) fn buffer(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        query: BufferQuery,
        args: &[Spanned<mir::Operand<'tcx>>],
        destination: &mir::Place<'tcx>,
        span: Span,
    ) -> Result<()> {
        let [receiver] = args else {
            return Err((span, "invalid buffer query arguments".into()));
        };
        let buffer = ir::Operand::local(self.buffer_parameter(&receiver.node, span)?);
        let destination = self.place(destination, span)?;
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
        }
        Ok(())
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

    /// The slice type of a read buffer parameter.
    pub(super) fn buffer_slice_type(&mut self, parameter: usize, span: Span) -> Result<ir::Type> {
        let ir::Type::Buffer { element, .. } = self.locals[parameter] else {
            return Err((span, "buffer slices require a buffer".into()));
        };
        Ok(self.module.types.intern_pointer(ir::Pointer {
            target: element.ty(),
            space: ir::AddressSpace::Device,
            mutable: false,
            slice: true,
        }))
    }
}
