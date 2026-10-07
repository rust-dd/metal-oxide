use metal_oxide_ir as ir;
use rustc_middle::mir;
use rustc_span::Span;

use super::{Result, function::FunctionImporter};

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(super) fn place(&self, place: &mir::Place<'tcx>, span: Span) -> Result<ir::Place> {
        let local = place.local.as_usize();
        let mut ty = self.locals[local];
        let mut projection = Vec::new();
        for element in place.projection.iter() {
            let index = match element {
                mir::ProjectionElem::Field(field, _) => field.as_u32(),
                mir::ProjectionElem::Index(index) => match self.constants[index.as_usize()] {
                    Some(ir::Constant::U32(value)) => value,
                    _ => return Err((span, "array indices must be provable constants".into())),
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
                    index
                        .and_then(|n| u32::try_from(n).ok())
                        .ok_or((span, "invalid constant array projection".into()))?
                }
                _ => return Err((span, "unsupported MIR place projection".into())),
            };
            ty = self
                .module
                .types
                .table()
                .field(ty, index)
                .ok_or((span, "invalid aggregate field or array index".into()))?;
            projection.push(index);
        }
        Ok(ir::Place { local, projection })
    }
}
