use metal_oxide_ir as ir;
use rustc_middle::mir;
use rustc_span::Span;

use super::{Result, function::FunctionImporter};

/// An imported place, what it denotes, and the address space it lies in.
pub(super) struct LocatedPlace {
    pub place: ir::Place,
    pub located: ir::Located,
    pub space: ir::AddressSpace,
}

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(super) fn place(&self, place: &mir::Place<'tcx>, span: Span) -> Result<ir::Place> {
        Ok(self.located_place(place, span)?.place)
    }

    pub(super) fn typed_place(
        &self,
        place: &mir::Place<'tcx>,
        span: Span,
    ) -> Result<(ir::Place, ir::Type)> {
        let LocatedPlace { place, located, .. } = self.located_place(place, span)?;
        match located {
            ir::Located::Value(ty) => Ok((place, ty)),
            ir::Located::Elements(_) => Err((span, "a slice is not one value".into())),
        }
    }

    pub(super) fn located_place(
        &self,
        place: &mir::Place<'tcx>,
        span: Span,
    ) -> Result<LocatedPlace> {
        let local = place.local.as_usize();
        if self.handles[local].is_some() {
            return Err((
                span,
                "references to ranges and device buffers are supported only as method receivers"
                    .into(),
            ));
        }
        let table = self.module.types.table();
        let mut located = ir::Located::Value(self.locals[local]);
        let mut space = ir::AddressSpace::Thread;
        let mut projection = Vec::new();
        let mut some = false;
        for element in place.projection.iter() {
            let array = match located {
                ir::Located::Value(ir::Type::Aggregate(id)) => match table.get(id) {
                    Some(ir::Aggregate::Array { length, .. }) => Some(*length),
                    _ => None,
                },
                _ => None,
            };
            let next = match element {
                // `Some`'s only field is the record's value; `None` has no fields to project.
                mir::ProjectionElem::Downcast(_, variant)
                    if variant.as_u32() == 1
                        && !some
                        && matches!(located, ir::Located::Value(ty) if self.module.types.is_option(ty)) =>
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
                mir::ProjectionElem::Deref => ir::Projection::Deref,
                mir::ProjectionElem::Index(index) => {
                    match (array, self.constants[index.as_usize()]) {
                        // Slice indices stay in their local, which the bounds check reads too.
                        (Some(_), Some(ir::Constant::Usize(value)))
                            if u32::try_from(value).is_ok() =>
                        {
                            ir::Projection::Field(value as u32)
                        }
                        _ => ir::Projection::Index(index.as_usize()),
                    }
                }
                mir::ProjectionElem::ConstantIndex {
                    offset,
                    min_length,
                    from_end,
                } => {
                    let length =
                        u64::from(array.ok_or((span, "slice patterns are unsupported".into()))?);
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
            located = match (next, located) {
                (ir::Projection::Field(field), ir::Located::Value(ty)) => {
                    table.field(ty, field).map(ir::Located::Value)
                }
                (ir::Projection::Index(_), ir::Located::Value(ir::Type::Aggregate(id))) => {
                    match table.get(id) {
                        Some(ir::Aggregate::Array { element, .. }) => {
                            Some(ir::Located::Value(*element))
                        }
                        _ => None,
                    }
                }
                (ir::Projection::Index(_), ir::Located::Elements(slice)) => {
                    Some(ir::Located::Value(slice.target))
                }
                (ir::Projection::Deref, ir::Located::Value(ty)) => {
                    table.pointer(ty).map(|pointer| {
                        space = pointer.space;
                        if pointer.slice {
                            ir::Located::Elements(pointer)
                        } else {
                            ir::Located::Value(pointer.target)
                        }
                    })
                }
                _ => None,
            }
            .ok_or((span, "invalid aggregate field, index or dereference".into()))?;
            projection.push(next);
        }
        if some {
            return Err((span, "unsupported MIR place projection".into()));
        }
        Ok(LocatedPlace {
            place: ir::Place { local, projection },
            located,
            space,
        })
    }
}
