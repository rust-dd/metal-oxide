use metal_oxide_ir as ir;
use rustc_middle::mir;
use rustc_span::Span;

use super::{Result, function::FunctionImporter};

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(super) fn place(&self, place: &mir::Place<'tcx>, span: Span) -> Result<ir::Place> {
        Ok(self.typed_place(place, span)?.0)
    }

    pub(super) fn typed_place(
        &self,
        place: &mir::Place<'tcx>,
        span: Span,
    ) -> Result<(ir::Place, ir::Type)> {
        let local = place.local.as_usize();
        if self.references[local].is_some() {
            return Err((
                span,
                "references are supported only as range iterator arguments".into(),
            ));
        }
        let mut ty = self.locals[local];
        let mut projection = Vec::new();
        let mut some = false;
        for element in place.projection.iter() {
            let index = match element {
                // `Some`'s only field is the record's value; `None` has no fields to project.
                mir::ProjectionElem::Downcast(_, variant)
                    if variant.as_u32() == 1 && !some && self.module.types.is_option(ty) =>
                {
                    some = true;
                    continue;
                }
                mir::ProjectionElem::Field(field, _) if some => {
                    if field.as_u32() != 0 {
                        return Err((span, "invalid Option field projection".into()));
                    }
                    some = false;
                    ir::Projection::Field(1)
                }
                mir::ProjectionElem::Field(field, _) => ir::Projection::Field(field.as_u32()),
                mir::ProjectionElem::Index(index) => match self.constants[index.as_usize()] {
                    Some(ir::Constant::Usize(value)) if u32::try_from(value).is_ok() => {
                        ir::Projection::Field(value as u32)
                    }
                    _ => ir::Projection::Index(index.as_usize()),
                },
                mir::ProjectionElem::ConstantIndex {
                    offset,
                    min_length,
                    from_end,
                } => {
                    let length = self.module.types.table().field_count(ty) as u64;
                    if length < min_length {
                        return Err((span, "invalid constant array projection".into()));
                    }
                    let index = if from_end {
                        length.checked_sub(offset)
                    } else {
                        Some(offset)
                    };
                    ir::Projection::Field(
                        index
                            .and_then(|n| u32::try_from(n).ok())
                            .ok_or((span, "invalid constant array projection".into()))?,
                    )
                }
                _ => return Err((span, "unsupported MIR place projection".into())),
            };
            let table = self.module.types.table();
            ty = match index {
                ir::Projection::Field(field) => table.field(ty, field),
                ir::Projection::Index(_) => match ty {
                    ir::Type::Aggregate(id) => match table.get(id) {
                        Some(ir::Aggregate::Array { element, .. }) => Some(*element),
                        _ => None,
                    },
                    _ => None,
                },
            }
            .ok_or((span, "invalid aggregate field or array index".into()))?;
            projection.push(index);
        }
        if some {
            return Err((span, "unsupported MIR place projection".into()));
        }
        Ok((ir::Place { local, projection }, ty))
    }
}
