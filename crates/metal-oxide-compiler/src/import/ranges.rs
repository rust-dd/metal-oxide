use metal_oxide_ir as ir;
use rustc_middle::{mir, ty};
use rustc_span::{Span, Spanned};

use super::{Result, control::integer, function::FunctionImporter, location};

/// Locals bound once to `&mut local`, directly or by reborrow; they stand for that local.
pub(super) fn references(body: &mir::Body<'_>) -> Vec<Option<mir::Local>> {
    let mut bindings = vec![0_u32; body.local_decls.len()];
    let mut references = vec![None; body.local_decls.len()];
    for statement in body.basic_blocks.iter().flat_map(|block| &block.statements) {
        let mir::StatementKind::Assign(assignment) = &statement.kind else {
            continue;
        };
        let (destination, value) = assignment.as_ref();
        let Some(local) = destination.as_local() else {
            continue;
        };
        bindings[local.as_usize()] += 1;
        if let mir::Rvalue::Ref(_, mir::BorrowKind::Mut { .. }, referent) = value {
            references[local.as_usize()] = match referent.as_ref() {
                mir::PlaceRef {
                    local,
                    projection: [],
                } => Some(local),
                mir::PlaceRef {
                    local,
                    projection: [mir::ProjectionElem::Deref],
                } => references[local.as_usize()],
                _ => None,
            };
        }
    }
    for (reference, count) in references.iter_mut().zip(bindings) {
        if count != 1 {
            *reference = None;
        }
    }
    references
}

/// Locals assigned an enum discriminant, which only `Option` flags provide.
pub(super) fn discriminants(body: &mir::Body<'_>) -> Vec<bool> {
    let mut discriminants = vec![false; body.local_decls.len()];
    for statement in body.basic_blocks.iter().flat_map(|block| &block.statements) {
        if let mir::StatementKind::Assign(assignment) = &statement.kind
            && let (destination, mir::Rvalue::Discriminant(_)) = assignment.as_ref()
            && let Some(local) = destination.as_local()
        {
            discriminants[local.as_usize()] = true;
        }
    }
    discriminants
}

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(super) fn is_range_next(&self, func: &mir::Operand<'tcx>, body: &mir::Body<'tcx>) -> bool {
        let tcx = self.module.tcx;
        let ty = self.normalize_type(func.ty(&body.local_decls, tcx));
        let ty::FnDef(definition, arguments) = *ty.kind() else {
            return false;
        };
        let arguments = tcx.instantiate_bound_regions_with_erased(arguments);
        crate::intrinsics::range_next(tcx, definition, arguments).is_some()
    }

    /// Expands `Range::next`: below `end` it yields `start` and advances it, otherwise `None`.
    /// Returns the terminator that replaces the call.
    pub(super) fn range_next(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        args: &[Spanned<mir::Operand<'tcx>>],
        destination: &mir::Place<'tcx>,
        target: usize,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Terminator> {
        let range = match args {
            [argument] => argument
                .node
                .place()
                .and_then(|place| place.as_local())
                .and_then(|local| self.references[local.as_usize()]),
            _ => None,
        }
        .ok_or((
            span,
            "range iteration requires a borrowed local range".into(),
        ))?
        .as_usize();
        let (option, option_type) = self.typed_place(destination, span)?;
        let table = self.module.types.table();
        let (Some(ir::Type::Scalar(element)), ir::Type::Aggregate(option_id), true) = (
            table.field(option_type, 1),
            option_type,
            self.module.types.is_option(option_type),
        ) else {
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
        let (Some(zero), Some(one)) = (integer(element, 0), integer(element, 1)) else {
            return Err((span, "range iteration requires an integer element".into()));
        };
        let condition = ir::Place::local(self.temporary(ir::Type::Scalar(ir::Scalar::Bool)));
        let source = location(self.module.tcx, span);
        let statement = |destination: &ir::Place, value| ir::Statement {
            destination: destination.clone(),
            value,
            source: source.clone(),
        };
        let yielded = |is_some, value| {
            statement(
                &option,
                ir::Expression::Aggregate {
                    ty: option_id,
                    fields: vec![ir::Operand::Constant(ir::Constant::Bool(is_some)), value],
                },
            )
        };
        statements.push(statement(
            &condition,
            ir::Expression::Binary(
                ir::BinaryOp::Lt,
                ir::Operand::Place(start.clone()),
                ir::Operand::Place(end),
            ),
        ));
        let some = vec![
            yielded(true, ir::Operand::Place(start.clone())),
            statement(
                &start,
                ir::Expression::Binary(
                    ir::BinaryOp::Add,
                    ir::Operand::Place(start.clone()),
                    ir::Operand::Constant(one),
                ),
            ),
        ];
        let none = vec![yielded(false, ir::Operand::Constant(zero))];
        Ok(ir::Terminator::Branch {
            condition: ir::Operand::Place(condition),
            then_block: self.append_block(body, some, target, &source),
            else_block: self.append_block(body, none, target, &source),
        })
    }

    fn append_block(
        &mut self,
        body: &mir::Body<'tcx>,
        statements: Vec<ir::Statement>,
        target: usize,
        source: &ir::SourceLocation,
    ) -> usize {
        self.appended.push(ir::Block {
            statements,
            terminator: ir::Terminator::Goto(target),
            source: source.clone(),
        });
        body.basic_blocks.len() + self.appended.len() - 1
    }

    fn temporary(&mut self, ty: ir::Type) -> usize {
        self.locals.push(ty);
        self.locals.len() - 1
    }
}
