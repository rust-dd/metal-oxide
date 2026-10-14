use metal_oxide_ir as ir;
use rustc_middle::{
    mir,
    ty::{Ty, TypingEnv},
};
use rustc_span::Span;

use super::{Result, function::FunctionImporter};

impl<'tcx> FunctionImporter<'_, 'tcx> {
    /// `destination = transmute(value)` from an integer into an enum that Rust stores in it,
    /// such as `Option<NonZero<T>>`: one niche value is the dataless variant, and any other is
    /// the payload of the other variant.
    pub(super) fn niche_transmute(
        &mut self,
        destination: ir::Place,
        ty: ir::Type,
        value: &mir::Operand<'tcx>,
        to: Ty<'tcx>,
        span: Span,
    ) -> Result<Vec<ir::Statement>> {
        let tcx = self.module.tcx;
        let unsupported = || {
            (
                span,
                "transmutes into enums need an integer niche".to_string(),
            )
        };
        let layout = tcx
            .layout_of(TypingEnv::fully_monomorphized().as_query_input(self.normalize_type(to)))
            .map_err(|_| unsupported())?;
        let rustc_abi::Variants::Multiple {
            tag_encoding:
                rustc_abi::TagEncoding::Niche {
                    untagged_variant,
                    niche_variants,
                    niche_start,
                },
            tag_field,
            ..
        } = &layout.variants
        else {
            return Err(unsupported());
        };
        let (ir::Type::Scalar(scalar), ir::Type::Aggregate(id)) =
            (self.operand_type(value, span)?, ty)
        else {
            return Err(unsupported());
        };
        let niche = niche_variants.start;
        let payload = match self.module.types.table().get(id) {
            Some(ir::Aggregate::Enum { variants, .. }) if variants.len() == 2 => {
                variants[untagged_variant.as_usize()].clone()
            }
            _ => return Err(unsupported()),
        };
        if niche != niche_variants.last
            || payload != [ir::Type::Scalar(scalar)]
            || layout.fields.offset(tag_field.as_usize()) != rustc_abi::Size::ZERO
        {
            return Err(unsupported());
        }
        let niche_value = super::control::integer(scalar, *niche_start).ok_or_else(unsupported)?;
        let tag = self.module.types.tag(ty).ok_or_else(unsupported)?;
        let operand = self.operand(value, span)?;
        let flag = self.temporary(ir::Type::Scalar(ir::Scalar::Bool));
        let mut tag_place = destination.clone();
        tag_place.projection.push(ir::Projection::Field(0));
        let source = super::location(tcx, span);
        let statement = |destination, value| ir::Statement {
            destination,
            value,
            source: source.clone(),
        };
        // The flag holds whether the tag is one.
        let comparison = if niche.as_u32() == 1 {
            ir::BinaryOp::Eq
        } else {
            ir::BinaryOp::Ne
        };
        Ok(vec![
            statement(
                destination,
                ir::Expression::Variant {
                    ty: id,
                    variant: untagged_variant.as_u32(),
                    fields: vec![operand.clone()],
                },
            ),
            statement(
                ir::Place::local(flag),
                ir::Expression::Binary(comparison, operand, ir::Operand::Constant(niche_value)),
            ),
            statement(
                tag_place,
                ir::Expression::Cast(ir::Operand::local(flag), tag),
            ),
        ])
    }

    /// Whether an aggregate builds a raw pointer or a restricted value, which hold their first
    /// field's value.
    pub(super) fn is_wrapping(&self, kind: &mir::AggregateKind<'tcx>) -> bool {
        let tcx = self.module.tcx;
        match *kind {
            mir::AggregateKind::RawPtr(..) => true,
            mir::AggregateKind::Adt(definition, _, arguments, ..) => crate::types::restricted(
                tcx,
                self.normalize_type(Ty::new_adt(tcx, tcx.adt_def(definition), arguments)),
            )
            .is_some(),
            _ => false,
        }
    }

    /// The IR type of an operand's value.
    pub(super) fn operand_type(
        &mut self,
        operand: &mir::Operand<'tcx>,
        span: Span,
    ) -> Result<ir::Type> {
        Ok(self
            .argument_type(operand, &[], span)?
            .expect("every local has a type once references are inferred"))
    }

    /// The pointer `value` as `to`, which must view the same values in the same space; pointees
    /// may differ in the mutability of pointers they hold.
    pub(super) fn pointer_cast(
        &mut self,
        value: &mir::Operand<'tcx>,
        to: ir::Type,
        span: Span,
    ) -> Result<ir::Expression> {
        let from = self.operand_type(value, span)?;
        let operand = self.operand(value, span)?;
        if from == to {
            return Ok(ir::Expression::Use(operand));
        }
        let table = self.module.types.table();
        match (table.pointer(from), table.pointer(to)) {
            (Some(from), Some(target))
                if table.same_values(from.target, target.target)
                    && from.space == target.space
                    && (from.slice || !target.slice) =>
            {
                Ok(ir::Expression::PointerCast(operand, to))
            }
            _ => Err((
                span,
                "pointer casts must keep the pointee type and address space".into(),
            )),
        }
    }
}
