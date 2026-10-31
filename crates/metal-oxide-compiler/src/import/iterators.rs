use metal_oxide_ir as ir;
use rustc_middle::mir;
use rustc_span::{Span, Spanned};

use super::{Result, calls::Callee, function::FunctionImporter};
use crate::intrinsics::Expanded;

/// The fields of `core::slice::ChunksExact` that `next` uses: the slice of chunks not yet
/// yielded and the chunk size.
const CHUNKS: u32 = 0;
const CHUNK_SIZE: u32 = 2;

impl<'tcx> FunctionImporter<'_, 'tcx> {
    /// The iterator whose `next` a call advances, and where it is: a local its receiver borrows,
    /// or for an array iterator any place its receiver points to.
    pub(super) fn expanded_next(
        &self,
        callee: &Callee<'tcx>,
        args: &[Spanned<mir::Operand<'tcx>>],
    ) -> Option<(Expanded, ir::Place)> {
        let expanded =
            crate::intrinsics::expanded_next(self.module.tcx, callee.definition, callee.arguments)?;
        if let Some(local) = self.receiver(args) {
            return Some((expanded, ir::Place::local(local)));
        }
        let pointer = args.first()?.node.place()?.as_local()?.as_usize();
        (expanded == Expanded::Array)
            .then(|| ir::Place {
                local: pointer,
                projection: vec![ir::Projection::Deref],
            })
            .map(|place| (expanded, place))
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
        let some = vec![
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
        let none = vec![yielded(Vec::new())];
        Ok(self.yield_arms(
            body,
            (destination, target),
            ir::Operand::local(available),
            (some, none),
            &source,
        ))
    }

    /// Expands `array::IntoIter::next`: below the array's length it yields the element at the
    /// iterator's index and advances the index, otherwise `None`.
    pub(super) fn array_next(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        iterator: ir::Place,
        destination: &mir::Place<'tcx>,
        target: usize,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Terminator> {
        let (option, option_type) = self.typed_place(destination, span)?;
        let table = self.module.types.table();
        let iterator_type = match iterator.projection.as_slice() {
            [] => Some(self.locals[iterator.local]),
            _ => table
                .pointer(self.locals[iterator.local])
                .map(|pointer| pointer.target),
        };
        let length = match iterator_type.and_then(|ty| table.field(ty, 0)) {
            Some(ir::Type::Aggregate(id)) => match table.get(id) {
                Some(ir::Aggregate::Array { length, .. }) => u64::from(*length),
                _ => return Err((span, "invalid array iterator".into())),
            },
            _ => return Err((span, "invalid array iterator".into())),
        };
        let ir::Type::Aggregate(option_id) = option_type else {
            return Err((span, "invalid array iterator result".into()));
        };
        let index = self.temporary(ir::Type::Scalar(ir::Scalar::Usize));
        let available = self.temporary(ir::Type::Scalar(ir::Scalar::Bool));
        let source = self.check_location(span);
        let statement = |destination: ir::Place, value| ir::Statement {
            destination,
            value,
            source: source.clone(),
        };
        let field = |fields: &[ir::Projection]| {
            let mut place = iterator.clone();
            place.projection.extend_from_slice(fields);
            place
        };
        let next = field(&[ir::Projection::Field(1)]);
        // The bound check reads a copy of the index, so the prover bounds the element access.
        statements.push(statement(
            ir::Place::local(index),
            ir::Expression::Use(ir::Operand::Place(next.clone())),
        ));
        statements.push(statement(
            ir::Place::local(available),
            ir::Expression::Binary(
                ir::BinaryOp::Lt,
                ir::Operand::local(index),
                ir::Operand::Constant(ir::Constant::Usize(length)),
            ),
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
        let element = field(&[ir::Projection::Field(0), ir::Projection::Index(index)]);
        let some = vec![
            yielded(vec![ir::Operand::Place(element)]),
            statement(
                next,
                ir::Expression::Binary(
                    ir::BinaryOp::Add,
                    ir::Operand::local(index),
                    ir::Operand::Constant(ir::Constant::Usize(1)),
                ),
            ),
        ];
        let none = vec![yielded(Vec::new())];
        Ok(self.yield_arms(
            body,
            (destination, target),
            ir::Operand::local(available),
            (some, none),
            &source,
        ))
    }
}
