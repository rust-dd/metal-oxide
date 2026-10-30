use metal_oxide_ir as ir;
use rustc_middle::mir;
use rustc_span::{Span, Spanned};

use super::{Result, function::FunctionImporter, location};

impl<'tcx> FunctionImporter<'_, 'tcx> {
    /// Exchanges two pointees through a temporary, which also holds when the pointers are equal.
    pub(super) fn swap(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        args: &[Spanned<mir::Operand<'tcx>>],
        span: Span,
    ) -> Result<()> {
        let [(x, _, ty), (y, _, _)] = [
            self.element_pointer(&args[0])?,
            self.element_pointer(&args[1])?,
        ];
        if ty == ir::Type::Unit {
            return Ok(());
        }
        let (x, y) = (deref(&x), deref(&y));
        let temporary = ir::Place::local(self.temporary(ty));
        let source = location(self.module.tcx, span);
        let statement = |destination, value| ir::Statement {
            destination,
            value,
            source: source.clone(),
        };
        statements.extend([
            statement(temporary.clone(), read(&x)),
            statement(x, read(&y)),
            statement(y, read(&temporary)),
        ]);
        Ok(())
    }

    /// Copies `count` elements one at a time; the current block enters the loop, which exits
    /// to `target`. Returns the current block's terminator.
    pub(super) fn copy_nonoverlapping(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        args: &[Spanned<mir::Operand<'tcx>>],
        target: usize,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Terminator> {
        let [from, to, count] = args else {
            return Err((span, "invalid copy arguments".into()));
        };
        let (from, from_type, ty) = self.element_pointer(from)?;
        let (to, to_type, _) = self.element_pointer(to)?;
        if ty == ir::Type::Unit {
            return Ok(ir::Terminator::Goto(target));
        }
        let count = self.operand(&count.node, count.span)?;
        let index = ir::Place::local(self.temporary(ir::Type::Scalar(ir::Scalar::Usize)));
        let condition = ir::Place::local(self.temporary(ir::Type::Scalar(ir::Scalar::Bool)));
        let (element, copy) = (
            ir::Place::local(self.temporary(from_type)),
            ir::Place::local(self.temporary(to_type)),
        );
        let source = location(self.module.tcx, span);
        let statement = |destination, value| ir::Statement {
            destination,
            value,
            source: source.clone(),
        };
        let at = |pointer: ir::Place| {
            ir::Expression::Binary(
                ir::BinaryOp::Offset,
                ir::Operand::Place(pointer),
                ir::Operand::Place(index.clone()),
            )
        };
        let usize = |value| ir::Operand::Constant(ir::Constant::Usize(value));
        statements.push(statement(index.clone(), ir::Expression::Use(usize(0))));
        let header = body.basic_blocks.len() + self.appended.len();
        let condition_value =
            ir::Expression::Binary(ir::BinaryOp::Lt, ir::Operand::Place(index.clone()), count);
        let iteration = vec![
            statement(element.clone(), at(from)),
            statement(copy.clone(), at(to)),
            statement(deref(&copy), read(&deref(&element))),
            statement(
                index.clone(),
                ir::Expression::Binary(
                    ir::BinaryOp::Add,
                    ir::Operand::Place(index.clone()),
                    usize(1),
                ),
            ),
        ];
        self.append(
            body,
            ir::Block {
                statements: vec![statement(condition.clone(), condition_value)],
                terminator: ir::Terminator::Branch {
                    condition: ir::Operand::Place(condition),
                    then_block: header + 1,
                    else_block: target,
                },
                source: source.clone(),
            },
        );
        self.append(
            body,
            ir::Block {
                statements: iteration,
                terminator: ir::Terminator::Goto(header),
                source,
            },
        );
        Ok(ir::Terminator::Goto(header))
    }

    /// A thin pointer argument's place, its type and its pointee's type.
    fn element_pointer(
        &mut self,
        argument: &Spanned<mir::Operand<'tcx>>,
    ) -> Result<(ir::Place, ir::Type, ir::Type)> {
        let span = argument.span;
        let ty = self.operand_type(&argument.node, span)?;
        let target = self
            .module
            .types
            .table()
            .pointer(ty)
            .filter(|pointer| !pointer.slice)
            .map(|pointer| pointer.target);
        match (self.operand(&argument.node, span)?, target) {
            (ir::Operand::Place(place), Some(target)) => Ok((place, ty, target)),
            _ => Err((span, "memory operations require element pointers".into())),
        }
    }
}

fn deref(pointer: &ir::Place) -> ir::Place {
    let mut place = pointer.clone();
    place.projection.push(ir::Projection::Deref);
    place
}

fn read(place: &ir::Place) -> ir::Expression {
    ir::Expression::Use(ir::Operand::Place(place.clone()))
}
