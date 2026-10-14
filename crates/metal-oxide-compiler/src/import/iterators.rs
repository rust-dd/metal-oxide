use metal_oxide_ir as ir;
use rustc_middle::mir;
use rustc_span::{Span, Spanned};

use super::{Result, calls::Callee, function::FunctionImporter, ranges::matched};
use crate::intrinsics::Expanded;

/// The fields of `core::slice::ChunksExact` that `next` uses: the slice of chunks not yet
/// yielded and the chunk size.
const CHUNKS: u32 = 0;
const CHUNK_SIZE: u32 = 2;

impl<'tcx> FunctionImporter<'_, 'tcx> {
    /// The iterator whose `next` a call advances, when its receiver borrows a local that the
    /// call can expand on.
    pub(super) fn expanded_next(
        &self,
        callee: &Callee<'tcx>,
        args: &[Spanned<mir::Operand<'tcx>>],
    ) -> Option<Expanded> {
        crate::intrinsics::expanded_next(self.module.tcx, callee.definition, callee.arguments)
            .filter(|_| self.receiver(args).is_some())
    }

    /// The local that a call's borrowed receiver stands for.
    pub(super) fn receiver(&self, args: &[Spanned<mir::Operand<'tcx>>]) -> Option<usize> {
        let local = args.first()?.node.place()?.as_local()?;
        Some(self.handles[local.as_usize()]?.as_usize())
    }

    /// The `Option` that `ChunksExact::next` returns, once the iterator's type is known.
    pub(super) fn chunk_type(
        &mut self,
        receiver: &mir::Operand<'tcx>,
        pending: &[bool],
        span: Span,
    ) -> Result<Option<ir::Type>> {
        let local = receiver
            .place()
            .and_then(|place| place.as_local())
            .and_then(|local| self.handles[local.as_usize()])
            .ok_or((
                span,
                "chunk iterators advance through a borrowed local".into(),
            ))?
            .as_usize();
        if pending.get(local).copied().unwrap_or(false) {
            return Ok(None);
        }
        let chunk = self
            .module
            .types
            .table()
            .field(self.locals[local], CHUNKS)
            .ok_or((span, "invalid chunk iterator".into()))?;
        Ok(Some(self.module.types.option(chunk)))
    }

    /// Expands `ChunksExact::next`: while a chunk is left, it yields the first one and drops it.
    pub(super) fn next_chunk(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        local: usize,
        destination: &mir::Place<'tcx>,
        target: usize,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Terminator> {
        let table = self.module.types.table();
        let chunk = table.field(self.locals[local], CHUNKS);
        let (option, option_type) = self.typed_place(destination, span)?;
        let ir::Type::Aggregate(option_id) = option_type else {
            return Err((span, "invalid chunk iterator result".into()));
        };
        let Some(chunk) =
            chunk.filter(|&chunk| self.module.types.option_value(option_type) == Some(chunk))
        else {
            return Err((span, "invalid chunk iterator result".into()));
        };
        let source = self.check_location(span);
        let statement = |destination: ir::Place, value| ir::Statement {
            destination,
            value,
            source: source.clone(),
        };
        let field = |index| ir::Place {
            local,
            projection: vec![ir::Projection::Field(index)],
        };
        let (rest, size) = (field(CHUNKS), ir::Operand::Place(field(CHUNK_SIZE)));
        let length = self.temporary(ir::Type::Scalar(ir::Scalar::Usize));
        let available = self.temporary(ir::Type::Scalar(ir::Scalar::Bool));
        let value = self.temporary(chunk);
        statements.push(statement(
            ir::Place::local(length),
            ir::Expression::Len(ir::Operand::Place(rest.clone())),
        ));
        statements.push(statement(
            ir::Place::local(available),
            ir::Expression::Binary(ir::BinaryOp::Le, size.clone(), ir::Operand::local(length)),
        ));
        let yielded = |fields: Vec<ir::Operand>| {
            statement(
                option.clone(),
                ir::Expression::Variant {
                    ty: option_id,
                    variant: fields.len() as u32,
                    fields,
                },
            )
        };
        let mut some = vec![
            statement(
                ir::Place::local(value),
                ir::Expression::Subslice {
                    slice: ir::Operand::Place(rest.clone()),
                    start: ir::Operand::Constant(ir::Constant::Usize(0)),
                    end: size.clone(),
                },
            ),
            statement(
                rest.clone(),
                ir::Expression::Subslice {
                    slice: ir::Operand::Place(rest),
                    start: size,
                    end: ir::Operand::local(length),
                },
            ),
            yielded(vec![ir::Operand::local(value)]),
        ];
        let mut none = vec![yielded(Vec::new())];
        // As for ranges, branching to the arms of a following `match` skips the flag test.
        let (some_target, none_target) = match matched(body, target, destination) {
            Some((discriminant, some_target, none_target)) => {
                let flag = ir::Place::local(discriminant.as_usize());
                for (statements, variant) in [(&mut some, 1), (&mut none, 0)] {
                    statements.push(statement(
                        flag.clone(),
                        ir::Expression::Use(ir::Operand::Constant(ir::Constant::U8(variant))),
                    ));
                }
                (some_target.as_usize(), none_target.as_usize())
            }
            None => (target, target),
        };
        let mut arm = |statements, target| {
            self.append(
                body,
                ir::Block {
                    statements,
                    terminator: ir::Terminator::Goto(target),
                    source: source.clone(),
                },
            )
        };
        let (then_block, else_block) = (arm(some, some_target), arm(none, none_target));
        Ok(ir::Terminator::Branch {
            condition: ir::Operand::local(available),
            then_block,
            else_block,
        })
    }
}
