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
        let [(x, ty), (y, _)] = [self.pointee(&args[0])?, self.pointee(&args[1])?];
        if ty == ir::Type::Unit {
            return Ok(());
        }
        let temporary = self.temporary(ty);
        let source = location(self.module.tcx, span);
        let read = |place: &ir::Place| ir::Expression::Use(ir::Operand::Place(place.clone()));
        statements.extend([
            ir::Statement {
                destination: ir::Place::local(temporary),
                value: read(&x),
                source: source.clone(),
            },
            ir::Statement {
                destination: x,
                value: read(&y),
                source: source.clone(),
            },
            ir::Statement {
                destination: y,
                value: ir::Expression::Use(ir::Operand::local(temporary)),
                source,
            },
        ]);
        Ok(())
    }

    /// The place a thin pointer argument points to, and its type.
    fn pointee(&mut self, argument: &Spanned<mir::Operand<'tcx>>) -> Result<(ir::Place, ir::Type)> {
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
            (ir::Operand::Place(mut place), Some(target)) => {
                place.projection.push(ir::Projection::Deref);
                Ok((place, target))
            }
            _ => Err((span, "a swap requires two element pointers".into())),
        }
    }
}
