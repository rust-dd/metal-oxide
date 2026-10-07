use metal_oxide_ir as ir;
use rustc_middle::ty::{self, Ty, TyCtxt, TypingEnv, consts::ConstExt};
use rustc_span::{Span, Symbol};

pub(crate) struct TypeLowering<'tcx> {
    tcx: TyCtxt<'tcx>,
    types: ir::TypeTable,
}

impl<'tcx> TypeLowering<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            types: ir::TypeTable::default(),
        }
    }

    pub(crate) fn table(&self) -> &ir::TypeTable {
        &self.types
    }

    pub(crate) fn into_types(self) -> ir::TypeTable {
        self.types
    }

    pub(crate) fn lower(&mut self, ty: Ty<'tcx>, span: Span) -> Result<ir::Type, (Span, String)> {
        if let Some(scalar) = scalar(ty) {
            return Ok(ir::Type::Scalar(scalar));
        }
        if ty.is_unit() {
            return Ok(ir::Type::Unit);
        }
        if ty.is_never() {
            return Ok(ir::Type::Never);
        }
        if let ty::Tuple(types) = ty.kind()
            && types.len() == 2
            && types[1].is_bool()
            && let Some(scalar) = scalar(types[0])
            && scalar.is_integer()
        {
            return Ok(ir::Type::Checked(scalar));
        }
        if let Some((access, element)) = buffer(self.tcx, ty) {
            let element = match self.owned(element, span)? {
                ir::Type::Scalar(s) if !matches!(s, ir::Scalar::Bool | ir::Scalar::Usize) => {
                    ir::Element::Scalar(s)
                }
                ir::Type::Aggregate(id) => ir::Element::Aggregate(id),
                _ => return Err((span, "unsupported buffer element type".into())),
            };
            return Ok(ir::Type::Buffer {
                element,
                access,
                address_space: if access == ir::Access::ReadWrite {
                    ir::AddressSpace::Threadgroup
                } else {
                    ir::AddressSpace::Device
                },
            });
        }
        if let ty::Array(element, length) = ty.kind() {
            let length = length
                .try_to_target_usize(self.tcx)
                .and_then(|n| u32::try_from(n).ok())
                .filter(|&n| n != 0)
                .ok_or((
                    span,
                    "array length must be a nonzero concrete u32-sized constant".into(),
                ))?;
            let element = self.owned(*element, span)?;
            return Ok(self.types.intern(ir::Aggregate::Array { element, length }));
        }
        if let ty::Tuple(fields) = ty.kind() {
            let fields = fields
                .iter()
                .map(|ty| self.owned(ty, span))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(self.types.intern(ir::Aggregate::Tuple(fields)));
        }
        if let ty::Adt(definition, args) = ty.kind() {
            if self
                .tcx
                .get_diagnostic_item(Symbol::intern("metal_oxide_dim3"))
                == Some(definition.did())
            {
                return Ok(ir::Type::Dim3);
            }
            if definition.is_struct() {
                if ty.needs_drop(self.tcx, TypingEnv::fully_monomorphized()) {
                    return Err((span, "destructors cannot be imported".into()));
                }
                let fields = definition
                    .non_enum_variant()
                    .fields
                    .iter()
                    .map(|field| {
                        let ty = self.tcx.normalize_erasing_regions(
                            TypingEnv::fully_monomorphized(),
                            field.ty(self.tcx, args),
                        );
                        Ok(ir::RecordField {
                            name: field.name.to_string(),
                            ty: self.owned(ty, span)?,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if fields.is_empty() {
                    return Err((span, "empty device records are unsupported".into()));
                }
                return Ok(self.types.intern(ir::Aggregate::Record {
                    name: self.tcx.item_name(definition.did()).to_string(),
                    fields,
                }));
            }
        }
        Err((span, format!("unsupported device type: {ty}")))
    }
    fn owned(&mut self, ty: Ty<'tcx>, span: Span) -> Result<ir::Type, (Span, String)> {
        let ty = self.lower(ty, span)?;
        if matches!(
            ty,
            ir::Type::Unit | ir::Type::Never | ir::Type::Buffer { .. }
        ) {
            return Err((span, "aggregate components must be owned values".into()));
        }
        Ok(ty)
    }
}

fn scalar(ty: Ty<'_>) -> Option<ir::Scalar> {
    match ty.kind() {
        ty::Bool => Some(ir::Scalar::Bool),
        ty::Float(ty::FloatTy::F32) => Some(ir::Scalar::F32),
        ty::Uint(ty::UintTy::U32) => Some(ir::Scalar::U32),
        ty::Uint(ty::UintTy::Usize) => Some(ir::Scalar::Usize),
        ty::Int(ty::IntTy::I32) => Some(ir::Scalar::I32),
        ty::Uint(ty::UintTy::U8) => Some(ir::Scalar::U8),
        ty::Uint(ty::UintTy::U16) => Some(ir::Scalar::U16),
        ty::Int(ty::IntTy::I8) => Some(ir::Scalar::I8),
        ty::Int(ty::IntTy::I16) => Some(ir::Scalar::I16),
        _ => None,
    }
}

fn buffer<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<(ir::Access, Ty<'tcx>)> {
    let ty::Adt(definition, args) = ty.kind() else {
        return None;
    };
    let access = [
        ("metal_oxide_read_buffer", ir::Access::Read),
        ("metal_oxide_write_buffer", ir::Access::Write),
        ("metal_oxide_atomic_buffer", ir::Access::Atomic),
        ("metal_oxide_threadgroup_buffer", ir::Access::ReadWrite),
    ]
    .into_iter()
    .find_map(|(name, access)| {
        (tcx.get_diagnostic_item(Symbol::intern(name)) == Some(definition.did())).then_some(access)
    })?;
    Some((access, args.type_at(0)))
}

pub(crate) fn parameter<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<ir::Type> {
    let mut lowering = TypeLowering::new(tcx);
    let ty = lowering.lower(ty, rustc_span::DUMMY_SP).ok()?;
    match ty {
        ir::Type::Buffer {
            element,
            access,
            address_space: ir::AddressSpace::Device,
        } if lowering.types.is_abi_value(element.ty())
            && (access != ir::Access::Atomic
                || matches!(
                    element,
                    ir::Element::Scalar(ir::Scalar::U32 | ir::Scalar::I32)
                )) =>
        {
            Some(ty)
        }
        _ if lowering.types.is_abi_value(ty) => Some(ty),
        _ => None,
    }
}
