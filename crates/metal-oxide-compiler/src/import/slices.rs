use metal_oxide_ir as ir;
use rustc_middle::{mir, ty};
use rustc_span::{Span, Spanned};

use super::{Result, function::FunctionImporter, location};
use crate::intrinsics::Intrinsic;

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(super) fn is_subslice(
        &self,
        func: &mir::Operand<'tcx>,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<bool> {
        Ok(matches!(
            self.callee(func, body, span)?.intrinsic,
            Some(Intrinsic::Subslice)
        ))
    }

    /// Expands `slice[range]` into Rust's range checks, in core's order, and the subslice.
    /// Returns the terminator of the current block; `span` is where the checks report.
    pub(super) fn subslice(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        args: &[Spanned<mir::Operand<'tcx>>],
        destination: &mir::Place<'tcx>,
        target: usize,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Terminator> {
        let [slice, range] = args else {
            return Err((span, "invalid slice index arguments".into()));
        };
        let tcx = self.module.tcx;
        let range_type = self.normalize_type(range.node.ty(&body.local_decls, tcx));
        let receiver_type = self
            .argument_type(&slice.node, &[], slice.span)?
            .expect("inferred references are known");
        let receiver = self.operand(&slice.node, slice.span)?;
        let range = self.operand(&range.node, range.span)?;
        let source = location(tcx, span);
        let statement = |destination: ir::Place, value| ir::Statement {
            destination,
            value,
            source: source.clone(),
        };
        // Arrays index by delegating to their slice.
        let slice = if self
            .module
            .types
            .table()
            .pointer(receiver_type)
            .is_some_and(|pointer| !pointer.slice)
        {
            let ty = self.array_slice(receiver_type, span)?;
            let slice = ir::Place::local(self.temporary(ty));
            statements.push(statement(slice.clone(), ir::Expression::Slice(receiver)));
            ir::Operand::Place(slice)
        } else {
            receiver
        };
        let length = ir::Operand::local(self.temporary(ir::Type::Scalar(ir::Scalar::Usize)));
        let ir::Operand::Place(length_place) = &length else {
            unreachable!()
        };
        statements.push(statement(
            length_place.clone(),
            ir::Expression::Len(slice.clone()),
        ));
        let zero = ir::Operand::Constant(ir::Constant::Usize(0));
        let items = tcx.lang_items();
        let ty::Adt(definition, arguments) = range_type.kind() else {
            return Err((span, "slices are indexed by usize ranges".into()));
        };
        let field = |name: &str| self.range_field(&range, range_type, name, span);
        let (start, end, checks) = if items.range_full_struct() == Some(definition.did()) {
            (zero, length.clone(), vec![])
        } else if !arguments.type_at(0).is_usize() {
            return Err((span, "slices are indexed by usize ranges".into()));
        } else if items.range_struct() == Some(definition.did()) {
            let (start, end) = (field("start")?, field("end")?);
            let checks = vec![
                (start.clone(), end.clone(), ir::Check::SliceOrder),
                (end.clone(), length.clone(), ir::Check::SliceEnd),
            ];
            (start, end, checks)
        } else if items.range_from_struct() == Some(definition.did()) {
            let start = field("start")?;
            let checks = vec![(start.clone(), length.clone(), ir::Check::SliceStart)];
            (start, length.clone(), checks)
        } else if items.range_to_struct() == Some(definition.did()) {
            let end = field("end")?;
            let checks = vec![(end.clone(), length.clone(), ir::Check::SliceEnd)];
            (zero, end, checks)
        } else {
            return Err((span, "slices are indexed by usize ranges".into()));
        };
        let subslice = statement(
            self.place(destination, span)?,
            ir::Expression::Subslice { slice, start, end },
        );
        let Some(((first_low, first_high, first_check), rest)) = checks.split_first() else {
            statements.push(subslice);
            return Ok(ir::Terminator::Goto(target));
        };
        let mut next = self.append(
            body,
            ir::Block {
                statements: vec![subslice],
                terminator: ir::Terminator::Goto(target),
                source: source.clone(),
            },
        );
        for (low, high, check) in rest.iter().rev() {
            let mut checked = Vec::new();
            let terminator = self.ordered(
                &mut checked,
                low.clone(),
                high.clone(),
                *check,
                next,
                &source,
            );
            next = self.append(
                body,
                ir::Block {
                    statements: checked,
                    terminator,
                    source: source.clone(),
                },
            );
        }
        Ok(self.ordered(
            statements,
            first_low.clone(),
            first_high.clone(),
            *first_check,
            next,
            &source,
        ))
    }

    /// Asserts `low <= high`, continuing at `target`.
    fn ordered(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        low: ir::Operand,
        high: ir::Operand,
        check: ir::Check,
        target: usize,
        source: &ir::SourceLocation,
    ) -> ir::Terminator {
        let condition = ir::Place::local(self.temporary(ir::Type::Scalar(ir::Scalar::Bool)));
        statements.push(ir::Statement {
            destination: condition.clone(),
            value: ir::Expression::Binary(ir::BinaryOp::Le, low, high),
            source: source.clone(),
        });
        ir::Terminator::Assert {
            condition: ir::Operand::Place(condition),
            expected: true,
            enabled: true,
            target,
            check,
        }
    }

    /// A field of a `Range`, `RangeFrom` or `RangeTo` operand.
    fn range_field(
        &self,
        range: &ir::Operand,
        range_type: ty::Ty<'tcx>,
        name: &str,
        span: Span,
    ) -> Result<ir::Operand> {
        let ty::Adt(definition, _) = range_type.kind() else {
            unreachable!()
        };
        let index = definition
            .non_enum_variant()
            .fields
            .iter()
            .position(|field| field.name.as_str() == name)
            .ok_or((span, "invalid slice range".into()))?;
        Ok(match range {
            ir::Operand::Place(place) => {
                let mut place = place.clone();
                place.projection.push(ir::Projection::Field(index as u32));
                ir::Operand::Place(place)
            }
            ir::Operand::AggregateConstant { fields, .. } => fields[index].clone(),
            ir::Operand::Constant(_) => return Err((span, "invalid slice range".into())),
        })
    }
}
