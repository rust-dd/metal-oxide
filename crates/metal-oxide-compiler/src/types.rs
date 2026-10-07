use metal_oxide_ir as ir;
use rustc_middle::ty::{self, Ty, TyCtxt, TypingEnv};
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
        if let Some(buffer) = buffer(self.tcx, ty) {
            return Ok(buffer);
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
                let fields = definition
                    .non_enum_variant()
                    .fields
                    .iter()
                    .map(|field| {
                        let ty = field.ty(self.tcx, args);
                        scalar(
                            self.tcx
                                .normalize_erasing_regions(TypingEnv::fully_monomorphized(), ty),
                        )
                    })
                    .collect::<Option<Vec<_>>>()
                    .ok_or((span, "device records require scalar fields".into()))?;
                if fields.is_empty() {
                    return Err((span, "empty device records are unsupported".into()));
                }
                return Ok(self.types.intern(ir::Aggregate::Record(
                    fields.into_iter().map(ir::Type::Scalar).collect(),
                )));
            }
        }
        Err((span, format!("unsupported device type: {ty}")))
    }
}

fn scalar(ty: Ty<'_>) -> Option<ir::Scalar> {
    match ty.kind() {
        ty::Bool => Some(ir::Scalar::Bool),
        ty::Float(ty::FloatTy::F32) => Some(ir::Scalar::F32),
        ty::Uint(ty::UintTy::U32) => Some(ir::Scalar::U32),
        ty::Int(ty::IntTy::I32) => Some(ir::Scalar::I32),
        ty::Uint(ty::UintTy::U8) => Some(ir::Scalar::U8),
        ty::Uint(ty::UintTy::U16) => Some(ir::Scalar::U16),
        _ => None,
    }
}

fn buffer<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<ir::Type> {
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
    let element = scalar(args.type_at(0))?;
    if element == ir::Scalar::Bool {
        return None;
    }
    Some(ir::Type::Buffer {
        element,
        access,
        address_space: if access == ir::Access::ReadWrite {
            ir::AddressSpace::Threadgroup
        } else {
            ir::AddressSpace::Device
        },
    })
}

pub(crate) fn parameter<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<ir::Type> {
    let ty = scalar(ty)
        .map(ir::Type::Scalar)
        .or_else(|| buffer(tcx, ty))?;
    match ty {
        ir::Type::Scalar(ir::Scalar::Bool)
        | ir::Type::Buffer {
            address_space: ir::AddressSpace::Threadgroup,
            ..
        }
        | ir::Type::Buffer {
            element: ir::Scalar::F32 | ir::Scalar::U8 | ir::Scalar::U16,
            access: ir::Access::Atomic,
            ..
        } => None,
        _ => Some(ty),
    }
}
