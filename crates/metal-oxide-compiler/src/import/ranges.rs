use metal_oxide_ir as ir;
use rustc_middle::mir;
use rustc_span::Span;

use super::{Result, control::integer, function::FunctionImporter, location};

/// The discriminant local and the `Some` and `None` targets when `target` only reads the
/// discriminant of `option` and switches on it.
pub(super) fn matched<'tcx>(
    body: &mir::Body<'tcx>,
    target: usize,
    option: &mir::Place<'tcx>,
) -> Option<(mir::Local, mir::BasicBlock, mir::BasicBlock)> {
    let block = &body.basic_blocks[mir::BasicBlock::from_usize(target)];
    let mut discriminant = None;
    for statement in &block.statements {
        match &statement.kind {
            mir::StatementKind::StorageLive(_)
            | mir::StatementKind::StorageDead(_)
            | mir::StatementKind::Nop => {}
            mir::StatementKind::Assign(assignment)
                if discriminant.is_none() && assignment.1 == mir::Rvalue::Discriminant(*option) =>
            {
                discriminant = Some(assignment.0.as_local()?);
            }
            _ => return None,
        }
    }
    let discriminant = discriminant?;
    let mir::TerminatorKind::SwitchInt { discr, targets } = &block.terminator().kind else {
        return None;
    };
    if discr.place()?.as_local()? != discriminant {
        return None;
    }
    let arm = |value| {
        targets
            .iter()
            .find(|&(case, _)| case == value)
            .map_or(targets.otherwise(), |(_, target)| target)
    };
    Some((discriminant, arm(1), arm(0)))
}

/// For each local assigned an enum discriminant, the enum place it reads.
pub(super) fn discriminants<'tcx>(body: &mir::Body<'tcx>) -> Vec<Option<mir::Place<'tcx>>> {
    let mut discriminants = vec![None; body.local_decls.len()];
    for statement in body.basic_blocks.iter().flat_map(|block| &block.statements) {
        if let mir::StatementKind::Assign(assignment) = &statement.kind
            && let (destination, mir::Rvalue::Discriminant(place)) = assignment.as_ref()
            && let Some(local) = destination.as_local()
        {
            discriminants[local.as_usize()].get_or_insert(*place);
        }
    }
    discriminants
}

impl<'tcx> FunctionImporter<'_, 'tcx> {
    /// Expands `Range::next`: below `end` it yields `start` and advances it, otherwise `None`.
    /// Returns the terminator that replaces the call.
    pub(super) fn range_next(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        range: usize,
        destination: &mir::Place<'tcx>,
        target: usize,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Terminator> {
        let (option, option_type) = self.typed_place(destination, span)?;
        let table = self.module.types.table();
        let (Some(ir::Type::Scalar(element)), ir::Type::Aggregate(option_id)) =
            (self.module.types.option_value(option_type), option_type)
        else {
            return Err((span, "invalid range iterator result".into()));
        };
        let ir::Type::Aggregate(range_id) = self.locals[range] else {
            return Err((span, "invalid range iterator state".into()));
        };
        let field = |name: &str| match table.get(range_id) {
            Some(ir::Aggregate::Record { fields, .. }) => {
                let index = fields.iter().position(|field| field.name == name)?;
                (fields[index].ty == ir::Type::Scalar(element)).then(|| ir::Place {
                    local: range,
                    projection: vec![ir::Projection::Field(index as u32)],
                })
            }
            _ => None,
        };
        let (Some(start), Some(end)) = (field("start"), field("end")) else {
            return Err((span, "invalid range iterator bounds".into()));
        };
        let Some(one) = integer(element, 1) else {
            return Err((span, "range iteration requires an integer element".into()));
        };
        // The bound check reads a scalar copy, so numeric proofs refine the yielded value.
        let current = ir::Place::local(self.temporary(ir::Type::Scalar(element)));
        let condition = ir::Place::local(self.temporary(ir::Type::Scalar(ir::Scalar::Bool)));
        let source = location(self.module.tcx, span);
        let statement = |destination: &ir::Place, value| ir::Statement {
            destination: destination.clone(),
            value,
            source: source.clone(),
        };
        let yielded = |fields: Vec<ir::Operand>| {
            statement(
                &option,
                ir::Expression::Variant {
                    ty: option_id,
                    variant: fields.len() as u32,
                    fields,
                },
            )
        };
        statements.push(statement(
            &current,
            ir::Expression::Use(ir::Operand::Place(start.clone())),
        ));
        statements.push(statement(
            &condition,
            ir::Expression::Binary(
                ir::BinaryOp::Lt,
                ir::Operand::Place(current.clone()),
                ir::Operand::Place(end),
            ),
        ));
        let some = vec![
            yielded(vec![ir::Operand::Place(current.clone())]),
            statement(
                &start,
                ir::Expression::Binary(
                    ir::BinaryOp::Add,
                    ir::Operand::Place(current),
                    ir::Operand::Constant(one),
                ),
            ),
        ];
        let none = vec![yielded(Vec::new())];
        Ok(self.yield_arms(
            body,
            (destination, target),
            ir::Operand::Place(condition),
            (some, none),
            &source,
        ))
    }

    /// Branches on `condition` to the arms that yield `Some` and `None` into `destination`. A
    /// `for` or `while let` then switches on the discriminant; the arms set it and continue at
    /// its arms directly, which keeps the loop as lean as a counted `while`.
    pub(super) fn yield_arms(
        &mut self,
        body: &mir::Body<'tcx>,
        (destination, target): (&mir::Place<'tcx>, usize),
        condition: ir::Operand,
        (mut some, mut none): (Vec<ir::Statement>, Vec<ir::Statement>),
        source: &ir::SourceLocation,
    ) -> ir::Terminator {
        let (some_target, none_target) = match matched(body, target, destination) {
            Some((discriminant, some_target, none_target)) => {
                let flag = ir::Place::local(discriminant.as_usize());
                for (statements, variant) in [(&mut some, 1), (&mut none, 0)] {
                    statements.push(ir::Statement {
                        destination: flag.clone(),
                        value: ir::Expression::Use(ir::Operand::Constant(ir::Constant::U8(
                            variant,
                        ))),
                        source: source.clone(),
                    });
                }
                (some_target.as_usize(), none_target.as_usize())
            }
            None => (target, target),
        };
        ir::Terminator::Branch {
            condition,
            then_block: self.append_block(body, some, some_target, source),
            else_block: self.append_block(body, none, none_target, source),
        }
    }

    fn append_block(
        &mut self,
        body: &mir::Body<'tcx>,
        statements: Vec<ir::Statement>,
        target: usize,
        source: &ir::SourceLocation,
    ) -> usize {
        self.append(
            body,
            ir::Block {
                statements,
                terminator: ir::Terminator::Goto(target),
                source: source.clone(),
            },
        )
    }

    pub(super) fn temporary(&mut self, ty: ir::Type) -> usize {
        self.locals.push(ty);
        self.locals.len() - 1
    }
}
