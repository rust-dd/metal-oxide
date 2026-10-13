use metal_oxide_ir as ir;
use rustc_middle::mir;
use rustc_span::{Span, Spanned};

use super::{Result, function::FunctionImporter, location, ranges::matched};
use crate::intrinsics::{Intrinsic, SliceIterator};

impl<'tcx> FunctionImporter<'_, 'tcx> {
    /// The slice iterator a call creates.
    pub(super) fn created_iterator(
        &self,
        func: &mir::Operand<'tcx>,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<Option<SliceIterator>> {
        Ok(match self.callee(func, body, span)?.intrinsic {
            Some(Intrinsic::SliceIterator(iterator)) => Some(iterator),
            _ => None,
        })
    }

    /// The slice iterator whose `next` a call invokes.
    pub(super) fn advanced_iterator(
        &self,
        func: &mir::Operand<'tcx>,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<Option<SliceIterator>> {
        let callee = self.callee(func, body, span)?;
        Ok(crate::intrinsics::slice_next(
            self.module.tcx,
            callee.definition,
            callee.arguments,
        ))
    }

    /// The record of an iterator over `slice`.
    pub(super) fn iterator_type(
        &mut self,
        iterator: SliceIterator,
        slice: ir::Type,
        span: Span,
    ) -> Result<ir::Type> {
        let rest = match (iterator, self.module.types.table().pointer(slice)) {
            (SliceIterator::Mutable, Some(pointer)) if pointer.slice && pointer.mutable => slice,
            (_, Some(pointer)) if pointer.slice => self.module.types.intern_pointer(ir::Pointer {
                mutable: false,
                ..pointer
            }),
            _ => return Err((span, "slice iterators require a slice".into())),
        };
        Ok(self.module.types.iterator(iterator, rest))
    }

    /// The `Option` that `next` returns on the iterator `receiver` borrows, once its type is known.
    pub(super) fn next_item_type(
        &mut self,
        iterator: SliceIterator,
        receiver: &mir::Operand<'tcx>,
        pending: &[bool],
        span: Span,
    ) -> Result<Option<ir::Type>> {
        let local = self.iterator_local(receiver, span)?;
        if pending.get(local).copied().unwrap_or(false) {
            return Ok(None);
        }
        let item = self.item_type(iterator, self.locals[local], span)?;
        Ok(Some(self.module.types.optional(item)))
    }

    fn item_type(
        &mut self,
        iterator: SliceIterator,
        record: ir::Type,
        span: Span,
    ) -> Result<ir::Type> {
        let rest = self
            .module
            .types
            .table()
            .field(record, 0)
            .and_then(|rest| self.module.types.table().pointer(rest))
            .ok_or((span, "invalid slice iterator state".into()))?;
        Ok(match iterator {
            SliceIterator::Chunks => self.module.types.intern_pointer(rest),
            SliceIterator::Shared | SliceIterator::Mutable => {
                self.module.types.intern_pointer(ir::Pointer {
                    slice: false,
                    ..rest
                })
            }
        })
    }

    /// The iterator local a `&mut iterator` receiver stands for.
    fn iterator_local(&self, receiver: &mir::Operand<'tcx>, span: Span) -> Result<usize> {
        receiver
            .place()
            .and_then(|place| place.as_local())
            .and_then(|local| self.handles[local.as_usize()])
            .map(|local| local.as_usize())
            .ok_or((
                span,
                "slice iterators advance through a borrowed local".into(),
            ))
    }

    /// Creates a slice iterator; `chunks_exact` first panics like core on a zero chunk size.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_iterator(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        iterator: SliceIterator,
        args: &[Spanned<mir::Operand<'tcx>>],
        destination: &mir::Place<'tcx>,
        target: usize,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Terminator> {
        let (destination, record) = self.typed_place(destination, span)?;
        let ir::Type::Aggregate(id) = record else {
            return Err((span, "invalid slice iterator".into()));
        };
        let mut fields = Vec::with_capacity(args.len());
        for argument in args {
            fields.push(self.operand(&argument.node, argument.span)?);
        }
        let source = location(self.module.tcx, span);
        let create = ir::Statement {
            destination,
            value: ir::Expression::Aggregate { ty: id, fields },
            source: source.clone(),
        };
        let ir::Expression::Aggregate { fields, .. } = &create.value else {
            unreachable!()
        };
        if iterator != SliceIterator::Chunks {
            statements.push(create);
            return Ok(ir::Terminator::Goto(target));
        }
        let size = fields[1].clone();
        let nonzero = self.temporary(ir::Type::Scalar(ir::Scalar::Bool));
        statements.push(ir::Statement {
            destination: ir::Place::local(nonzero),
            value: ir::Expression::Binary(
                ir::BinaryOp::Ne,
                size,
                ir::Operand::Constant(ir::Constant::Usize(0)),
            ),
            source: source.clone(),
        });
        let then_block = self.append(
            body,
            ir::Block {
                statements: vec![create],
                terminator: ir::Terminator::Goto(target),
                source: source.clone(),
            },
        );
        let else_block = self.append(
            body,
            ir::Block {
                statements: Vec::new(),
                terminator: ir::Terminator::Panic("chunk size must be non-zero".into()),
                source,
            },
        );
        Ok(ir::Terminator::Branch {
            condition: ir::Operand::local(nonzero),
            then_block,
            else_block,
        })
    }

    /// Expands `next`: the first element (or chunk) of the remaining slice, which then drops it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn next_item(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        iterator: SliceIterator,
        args: &[Spanned<mir::Operand<'tcx>>],
        destination: &mir::Place<'tcx>,
        target: usize,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Terminator> {
        let [receiver] = args else {
            return Err((span, "invalid iterator arguments".into()));
        };
        let local = self.iterator_local(&receiver.node, span)?;
        let item = self.item_type(iterator, self.locals[local], span)?;
        let (option, option_type) = self.typed_place(destination, span)?;
        let (ir::Type::Aggregate(option_id), ir::Type::Pointer(null)) = (option_type, item) else {
            return Err((span, "invalid iterator result".into()));
        };
        let source = location(self.module.tcx, span);
        let statement = |destination: ir::Place, value| ir::Statement {
            destination,
            value,
            source: source.clone(),
        };
        let field = |index| ir::Place {
            local,
            projection: vec![ir::Projection::Field(index)],
        };
        let rest = field(0);
        let length = self.temporary(ir::Type::Scalar(ir::Scalar::Usize));
        let available = self.temporary(ir::Type::Scalar(ir::Scalar::Bool));
        let usize = |value| ir::Operand::Constant(ir::Constant::Usize(value));
        statements.push(statement(
            ir::Place::local(length),
            ir::Expression::Len(ir::Operand::Place(rest.clone())),
        ));
        let (step, condition) = match iterator {
            SliceIterator::Chunks => {
                let size = ir::Operand::Place(field(1));
                (
                    size.clone(),
                    ir::Expression::Binary(ir::BinaryOp::Le, size, ir::Operand::local(length)),
                )
            }
            SliceIterator::Shared | SliceIterator::Mutable => (
                usize(1),
                ir::Expression::Binary(ir::BinaryOp::Lt, usize(0), ir::Operand::local(length)),
            ),
        };
        statements.push(statement(ir::Place::local(available), condition));
        let value = self.temporary(item);
        let mut some = match iterator {
            SliceIterator::Chunks => vec![statement(
                ir::Place::local(value),
                ir::Expression::Subslice {
                    slice: ir::Operand::Place(rest.clone()),
                    start: usize(0),
                    end: step.clone(),
                },
            )],
            SliceIterator::Shared | SliceIterator::Mutable => {
                let first = self.temporary(ir::Type::Scalar(ir::Scalar::Usize));
                vec![
                    statement(ir::Place::local(first), ir::Expression::Use(usize(0))),
                    statement(
                        ir::Place::local(value),
                        ir::Expression::Ref {
                            place: ir::Operand::Place(ir::Place {
                                local,
                                projection: vec![
                                    ir::Projection::Field(0),
                                    ir::Projection::Deref,
                                    ir::Projection::Index(first),
                                ],
                            }),
                            mutable: iterator == SliceIterator::Mutable,
                        },
                    ),
                ]
            }
        };
        some.push(statement(
            rest.clone(),
            ir::Expression::Subslice {
                slice: ir::Operand::Place(rest),
                start: step,
                end: ir::Operand::local(length),
            },
        ));
        let yielded = |is_some, value| {
            statement(
                option.clone(),
                ir::Expression::Aggregate {
                    ty: option_id,
                    fields: vec![ir::Operand::Constant(ir::Constant::Bool(is_some)), value],
                },
            )
        };
        some.push(yielded(true, ir::Operand::local(value)));
        let mut none = vec![yielded(
            false,
            ir::Operand::Constant(ir::Constant::Null(null)),
        )];
        // As for ranges, branching to the arms of a following `match` skips the flag test.
        let (some_target, none_target) = match matched(body, target, destination) {
            Some((discriminant, some_target, none_target)) => {
                let flag = ir::Place::local(discriminant.as_usize());
                some.push(statement(
                    flag.clone(),
                    ir::Expression::Use(ir::Operand::Constant(ir::Constant::Bool(true))),
                ));
                none.push(statement(
                    flag,
                    ir::Expression::Use(ir::Operand::Constant(ir::Constant::Bool(false))),
                ));
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
