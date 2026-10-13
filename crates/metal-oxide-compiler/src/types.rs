use std::collections::HashSet;

use metal_oxide_ir as ir;
use rustc_middle::ty::{self, Ty, TyCtxt, TypingEnv, consts::ConstExt};
use rustc_span::Span;

use crate::intrinsics::{DeviceType, SliceIterator, device_type};

pub(crate) struct TypeLowering<'tcx> {
    tcx: TyCtxt<'tcx>,
    types: ir::TypeTable,
    options: HashSet<usize>,
}

impl<'tcx> TypeLowering<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            types: ir::TypeTable::default(),
            options: HashSet::new(),
        }
    }

    /// Whether `ty` is the internal `Option` record of range iteration.
    pub(crate) fn is_option(&self, ty: ir::Type) -> bool {
        matches!(ty, ir::Type::Aggregate(id) if self.options.contains(&id))
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
        if let Some((access, address_space, element)) = buffer(self.tcx, ty) {
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
                address_space,
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
            // `slice[..]` passes a field-less `RangeFull`, which needs no storage.
            if self.tcx.lang_items().range_full_struct() == Some(definition.did()) {
                return Ok(ir::Type::Unit);
            }
            if device_type(self.tcx, definition.did()) == Some(DeviceType::F16) {
                return Ok(ir::Type::Scalar(ir::Scalar::F16));
            }
            if device_type(self.tcx, definition.did()) == Some(DeviceType::Dim3) {
                return Ok(ir::Type::Dim3);
            }
            if self.tcx.lang_items().option_type() == Some(definition.did())
                && let Some(element) = scalar(args.type_at(0)).filter(|s| s.is_integer())
            {
                return Ok(self.option(ir::Type::Scalar(element)));
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

    /// `Option<T>` as an `is_some` flag and a value that is meaningful only when it is set.
    fn option(&mut self, value: ir::Type) -> ir::Type {
        let ty = self.types.intern(ir::Aggregate::Record {
            name: "Option".into(),
            fields: vec![
                ir::RecordField {
                    name: "is_some".into(),
                    ty: ir::Type::Scalar(ir::Scalar::Bool),
                },
                ir::RecordField {
                    name: "value".into(),
                    ty: value,
                },
            ],
        });
        if let ir::Type::Aggregate(id) = ty {
            self.options.insert(id);
        }
        ty
    }

    /// A reference or slice reference into `space`, or an `Option` of one; the target must be
    /// an owned value.
    pub(crate) fn reference(
        &mut self,
        ty: Ty<'tcx>,
        space: ir::AddressSpace,
        span: Span,
    ) -> Result<ir::Type, (Span, String)> {
        if let ty::Adt(definition, arguments) = ty.kind()
            && self.tcx.lang_items().option_type() == Some(definition.did())
        {
            let value = self.reference(arguments.type_at(0), space, span)?;
            return Ok(self.option(value));
        }
        if let Some(iterator) = SliceIterator::of(self.tcx, ty) {
            let ty::Adt(_, arguments) = ty.kind() else {
                unreachable!()
            };
            let element = arguments
                .types()
                .next()
                .ok_or((span, format!("unsupported device type: {ty}")))?;
            let mutability = if iterator == SliceIterator::Mutable {
                ty::Mutability::Mut
            } else {
                ty::Mutability::Not
            };
            let slice = Ty::new_ref(
                self.tcx,
                self.tcx.lifetimes.re_erased,
                Ty::new_slice(self.tcx, element),
                mutability,
            );
            let rest = self.reference(slice, space, span)?;
            return Ok(self.iterator(iterator, rest));
        }
        let (ty::Ref(_, target, mutability) | ty::RawPtr(target, mutability)) = *ty.kind() else {
            return Err((span, format!("unsupported device type: {ty}")));
        };
        let (target, slice) = match target.kind() {
            ty::Slice(element) => (*element, true),
            _ => (target, false),
        };
        let target = self.owned(target, span)?;
        Ok(self.types.intern_pointer(ir::Pointer {
            target,
            space,
            mutable: mutability.is_mut(),
            slice,
        }))
    }

    pub(crate) fn intern_pointer(&mut self, pointer: ir::Pointer) -> ir::Type {
        self.types.intern_pointer(pointer)
    }

    pub(crate) fn optional(&mut self, value: ir::Type) -> ir::Type {
        self.option(value)
    }

    /// A slice iterator: the slice it has not yielded and, for chunks, the chunk size.
    pub(crate) fn iterator(&mut self, iterator: SliceIterator, rest: ir::Type) -> ir::Type {
        let field = |name: &str, ty| ir::RecordField {
            name: name.into(),
            ty,
        };
        let (name, fields) = match iterator {
            SliceIterator::Shared => ("Iter", vec![field("rest", rest)]),
            SliceIterator::Mutable => ("IterMut", vec![field("rest", rest)]),
            SliceIterator::Chunks => (
                "ChunksExact",
                vec![
                    field("rest", rest),
                    field("size", ir::Type::Scalar(ir::Scalar::Usize)),
                ],
            ),
        };
        self.types.intern(ir::Aggregate::Record {
            name: name.into(),
            fields,
        })
    }

    /// The address space of a reference, or of the first reference a record holds.
    pub(crate) fn reference_space(&self, ty: ir::Type) -> Option<ir::AddressSpace> {
        if let Some(pointer) = self.types.pointer(ty) {
            return Some(pointer.space);
        }
        let ir::Type::Aggregate(id) = ty else {
            return None;
        };
        self.types
            .get(id)?
            .component_types()
            .find_map(|component| self.reference_space(component))
    }

    fn owned(&mut self, ty: Ty<'tcx>, span: Span) -> Result<ir::Type, (Span, String)> {
        if matches!(ty.kind(), ty::Ref(..) | ty::RawPtr(..)) {
            return Err((
                span,
                "references cannot be stored in records, tuples or arrays".into(),
            ));
        }
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
        ty::Uint(ty::UintTy::U64) => Some(ir::Scalar::U64),
        ty::Uint(ty::UintTy::Usize) => Some(ir::Scalar::Usize),
        ty::Int(ty::IntTy::I32) => Some(ir::Scalar::I32),
        ty::Uint(ty::UintTy::U8) => Some(ir::Scalar::U8),
        ty::Uint(ty::UintTy::U16) => Some(ir::Scalar::U16),
        ty::Int(ty::IntTy::I8) => Some(ir::Scalar::I8),
        ty::Int(ty::IntTy::I16) => Some(ir::Scalar::I16),
        _ => None,
    }
}

fn buffer<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<(ir::Access, ir::AddressSpace, Ty<'tcx>)> {
    let ty::Adt(definition, args) = ty.kind() else {
        return None;
    };
    let kind = device_type(tcx, definition.did())?;
    let access = match kind {
        DeviceType::ReadBuffer => ir::Access::Read,
        DeviceType::WriteBuffer | DeviceType::DisjointBuffer => ir::Access::Write,
        DeviceType::AtomicBuffer | DeviceType::AtomicThreadgroupBuffer => ir::Access::Atomic,
        DeviceType::ThreadgroupBuffer
        | DeviceType::DynamicThreadgroupBuffer
        | DeviceType::ReadWriteBuffer => ir::Access::ReadWrite,
        _ => return None,
    };
    let space = if matches!(
        kind,
        DeviceType::ThreadgroupBuffer
            | DeviceType::DynamicThreadgroupBuffer
            | DeviceType::AtomicThreadgroupBuffer
    ) {
        ir::AddressSpace::Threadgroup
    } else {
        ir::AddressSpace::Device
    };
    Some((access, space, args.first()?.as_type()?))
}

pub(crate) fn parameter<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<ir::Type> {
    let dynamic_shared = matches!(ty.kind(), ty::Adt(definition, _) if device_type(tcx, definition.did()) == Some(DeviceType::DynamicThreadgroupBuffer));
    let mut lowering = TypeLowering::new(tcx);
    let ty = lowering.lower(ty, rustc_span::DUMMY_SP).ok()?;
    match ty {
        ir::Type::Buffer {
            element,
            access: ir::Access::ReadWrite,
            address_space: ir::AddressSpace::Threadgroup,
        } if dynamic_shared && lowering.types.is_abi_value(element.ty()) => Some(ty),
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

/// Whether `ty` holds a reference, raw pointer or slice iterator, whose address space its uses
/// determine.
pub(crate) fn has_reference<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    ty.walk().any(|argument| {
        argument.as_type().is_some_and(|ty| {
            matches!(ty.kind(), ty::Ref(..) | ty::RawPtr(..))
                || SliceIterator::of(tcx, ty).is_some()
        })
    })
}
