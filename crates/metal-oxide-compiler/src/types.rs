use metal_oxide_ir as ir;
use rustc_middle::ty::{self, Ty, TyCtxt, TypingEnv, consts::ConstExt};
use rustc_span::Span;

use crate::intrinsics::{DeviceType, device_type};

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
        if ty.is_privately_uninhabited(self.tcx, TypingEnv::fully_monomorphized()) {
            return Ok(ir::Type::Never);
        }
        // `PhantomData`, fn items and closures without captures need no storage, nor do
        // references to them. Enums keep their tag; zero-length arrays are rejected below.
        if (!ty.is_enum() && !matches!(ty.kind(), ty::Array(..)) && is_zero_sized(self.tcx, ty))
            || addresses_nothing(self.tcx, ty)
        {
            return Ok(ir::Type::Unit);
        }
        if let Some(base) = restricted(self.tcx, ty) {
            return self.lower(base, span);
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
                ir::Type::Scalar(s)
                    if !matches!(s, ir::Scalar::Bool | ir::Scalar::Usize | ir::Scalar::Isize) =>
                {
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
            let length = self.array_length(*length, span)?;
            let element = self.owned(*element, span)?;
            if element == ir::Type::Unit {
                return Err((span, "arrays of zero-sized values are unsupported".into()));
            }
            return Ok(self.types.intern(ir::Aggregate::Array { element, length }));
        }
        if let ty::Tuple(fields) = ty.kind() {
            let fields = fields
                .iter()
                .map(|ty| self.owned(ty, span))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(self.types.intern(ir::Aggregate::Tuple(fields)));
        }
        if let ty::Closure(_, arguments) = ty.kind() {
            return self.closure(
                arguments.as_closure().upvar_tys(),
                &mut std::iter::empty(),
                span,
            );
        }
        if let ty::Adt(definition, args) = ty.kind() {
            if device_type(self.tcx, definition.did()) == Some(DeviceType::F16) {
                return Ok(ir::Type::Scalar(ir::Scalar::F16));
            }
            if device_type(self.tcx, definition.did()) == Some(DeviceType::Dim3) {
                return Ok(ir::Type::Dim3);
            }
            if definition.is_enum() {
                return self.enumeration(ty, &mut std::iter::empty(), span);
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
                return Ok(self.types.intern(ir::Aggregate::Record {
                    name: self.tcx.item_name(definition.did()).to_string(),
                    fields,
                }));
            }
        }
        Err((span, format!("unsupported device type: {ty}")))
    }

    fn array_length(&self, length: ty::Const<'tcx>, span: Span) -> Result<u32, (Span, String)> {
        length
            .try_to_target_usize(self.tcx)
            .and_then(|n| u32::try_from(n).ok())
            .filter(|&n| n != 0)
            .ok_or((
                span,
                "array length must be a nonzero concrete u32-sized constant".into(),
            ))
    }

    /// The tag type of an enum type.
    pub(crate) fn tag(&self, ty: ir::Type) -> Option<ir::Scalar> {
        let ir::Type::Aggregate(id) = ty else {
            return None;
        };
        self.types.get(id)?.tag()
    }

    /// The `T` of an `Option<T>` type.
    pub(crate) fn option_value(&self, ty: ir::Type) -> Option<ir::Type> {
        let ir::Type::Aggregate(id) = ty else {
            return None;
        };
        match self.types.get(id)? {
            ir::Aggregate::Enum { name, variants, .. }
                if name == "Option" && variants.len() == 2 && variants[0].is_empty() =>
            {
                variants[1]
                    .first()
                    .copied()
                    .filter(|_| variants[1].len() == 1)
            }
            _ => None,
        }
    }

    /// `Option<T>`: the core enum, whose `None` comes first.
    pub(crate) fn option(&mut self, value: ir::Type) -> ir::Type {
        self.types.intern(ir::Aggregate::enumeration(
            "Option",
            vec![Vec::new(), vec![value]],
        ))
    }

    /// A field-holding enum; references in its fields take spaces from `spaces`. Variants keep
    /// their declaration order, so a discriminant is its variant's index.
    fn enumeration(
        &mut self,
        ty: Ty<'tcx>,
        spaces: &mut dyn Iterator<Item = ir::AddressSpace>,
        span: Span,
    ) -> Result<ir::Type, (Span, String)> {
        let ty::Adt(definition, arguments) = ty.kind() else {
            unreachable!("only enum types are enumerations")
        };
        if definition
            .discriminants(self.tcx)
            .any(|(index, discriminant)| discriminant.val != index.as_u32() as u128)
        {
            return Err((
                span,
                format!("explicit enum discriminants are unsupported: {ty}"),
            ));
        }
        if ty.needs_drop(self.tcx, TypingEnv::fully_monomorphized()) {
            return Err((span, "destructors cannot be imported".into()));
        }
        let mut variants = Vec::with_capacity(definition.variants().len());
        for variant in definition.variants() {
            let fields = variant
                .fields
                .iter()
                .map(|field| {
                    self.tcx.normalize_erasing_regions(
                        TypingEnv::fully_monomorphized(),
                        field.ty(self.tcx, arguments),
                    )
                })
                .collect::<Vec<_>>();
            // A variant with an uninhabited field never exists, so it needs no payload.
            if fields.iter().any(|field| {
                field.is_privately_uninhabited(self.tcx, TypingEnv::fully_monomorphized())
            }) {
                variants.push(Vec::new());
                continue;
            }
            variants.push(
                fields
                    .into_iter()
                    .map(|field| self.with_spaces(field, spaces, span))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        Ok(self.types.intern(ir::Aggregate::enumeration(
            self.tcx.item_name(definition.did()).to_string(),
            variants,
        )))
    }

    /// `ty` with the address spaces of its references, taken in the order a walk over its
    /// components meets them: a reference before its target, fields in order, an array's
    /// element once.
    pub(crate) fn with_spaces(
        &mut self,
        ty: Ty<'tcx>,
        spaces: &mut dyn Iterator<Item = ir::AddressSpace>,
        span: Span,
    ) -> Result<ir::Type, (Span, String)> {
        if !has_reference(self.tcx, ty) {
            return self.owned(ty, span);
        }
        if let Some(base) = restricted(self.tcx, ty) {
            return self.with_spaces(base, spaces, span);
        }
        let mut space = || {
            spaces
                .next()
                .ok_or((span, format!("unsupported reference type: {ty}")))
        };
        match *ty.kind() {
            ty::Ref(_, target, mutability) | ty::RawPtr(target, mutability) => {
                let space = space()?;
                let (target, slice) = match target.kind() {
                    ty::Slice(element) => (*element, true),
                    _ => (target, false),
                };
                let target = self.with_spaces(target, spaces, span)?;
                return Ok(self.types.intern_pointer(ir::Pointer {
                    target,
                    space,
                    mutable: mutability.is_mut(),
                    slice,
                }));
            }
            ty::Tuple(fields) => {
                let fields = fields
                    .iter()
                    .map(|field| self.with_spaces(field, spaces, span))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(self.types.intern(ir::Aggregate::Tuple(fields)));
            }
            ty::Array(element, length) => {
                let length = self.array_length(length, span)?;
                let element = self.with_spaces(element, spaces, span)?;
                return Ok(self.types.intern(ir::Aggregate::Array { element, length }));
            }
            ty::Closure(_, arguments) => {
                return self.closure(arguments.as_closure().upvar_tys(), spaces, span);
            }
            _ => {}
        }
        let ty::Adt(definition, arguments) = ty.kind() else {
            return Err((span, format!("unsupported reference type: {ty}")));
        };
        if definition.is_enum() {
            return self.enumeration(ty, spaces, span);
        }
        if !definition.is_struct() {
            return Err((span, format!("unsupported reference type: {ty}")));
        }
        let fields = definition
            .non_enum_variant()
            .fields
            .iter()
            .map(|field| {
                let ty = self.tcx.normalize_erasing_regions(
                    TypingEnv::fully_monomorphized(),
                    field.ty(self.tcx, arguments),
                );
                Ok(ir::RecordField {
                    name: field.name.to_string(),
                    ty: self.with_spaces(ty, spaces, span)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.types.intern(ir::Aggregate::Record {
            name: self.tcx.item_name(definition.did()).to_string(),
            fields,
        }))
    }

    /// A closure's captures as a record; captured references take spaces from `spaces`.
    fn closure(
        &mut self,
        upvars: &'tcx ty::List<Ty<'tcx>>,
        spaces: &mut dyn Iterator<Item = ir::AddressSpace>,
        span: Span,
    ) -> Result<ir::Type, (Span, String)> {
        let fields = upvars
            .iter()
            .map(|upvar| self.with_spaces(upvar, spaces, span))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.types.intern(ir::Aggregate::record("closure", fields)))
    }

    /// The address spaces of the references in `ty`, in the order of `with_spaces`; an enum's
    /// fields count in variant order.
    pub(crate) fn spaces(&self, ty: ir::Type, spaces: &mut Vec<ir::AddressSpace>) {
        if let Some(pointer) = self.types.pointer(ty) {
            spaces.push(pointer.space);
            return self.spaces(pointer.target, spaces);
        }
        let ir::Type::Aggregate(id) = ty else {
            return;
        };
        match self.types.get(id) {
            Some(ir::Aggregate::Enum { variants, .. }) => {
                for &field in variants.iter().flatten() {
                    self.spaces(field, spaces);
                }
            }
            Some(aggregate) => {
                for component in aggregate.component_types() {
                    self.spaces(component, spaces);
                }
            }
            None => {}
        }
    }

    pub(crate) fn intern_pointer(&mut self, pointer: ir::Pointer) -> ir::Type {
        self.types.intern_pointer(pointer)
    }

    pub(crate) fn intern(&mut self, aggregate: ir::Aggregate) -> ir::Type {
        self.types.intern(aggregate)
    }

    fn owned(&mut self, ty: Ty<'tcx>, span: Span) -> Result<ir::Type, (Span, String)> {
        if has_reference(self.tcx, ty) {
            return Err((span, format!("unsupported type with references: {ty}")));
        }
        let ty = self.lower(ty, span)?;
        if matches!(ty, ir::Type::Never | ir::Type::Buffer { .. }) {
            return Err((span, "aggregate components must be owned values".into()));
        }
        Ok(ty)
    }
}

/// The type whose values `ty` restricts, when it is stored as one: a pattern type such as
/// `NonNull`'s non-null pointer, or a transparent struct around one, such as `NonZero<T>`.
pub(crate) fn restricted<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    match ty.kind() {
        ty::Pat(base, _) => Some(*base),
        ty::Adt(definition, arguments) if definition.repr().transparent() => {
            let mut fields = definition.non_enum_variant().fields.iter().map(|field| {
                tcx.normalize_erasing_regions(
                    TypingEnv::fully_monomorphized(),
                    field.ty(tcx, arguments),
                )
            });
            let field = fields.find(|field| !is_zero_sized(tcx, *field))?;
            restricted(tcx, field)
        }
        _ => None,
    }
}

/// The type a reference or raw pointer points to.
fn pointee(ty: Ty<'_>) -> Option<Ty<'_>> {
    match ty.kind() {
        ty::Ref(_, target, _) | ty::RawPtr(target, _) => Some(*target),
        _ => None,
    }
}

/// Whether `ty` is a reference or raw pointer to a zero-sized value, directly or through further
/// such references, which the IR stores nowhere; it needs no address.
fn addresses_nothing<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    pointee(ty).is_some_and(|target| is_zero_sized(tcx, target) || addresses_nothing(tcx, target))
}

/// Whether values of `ty` occupy no storage.
pub(crate) fn is_zero_sized<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    tcx.layout_of(TypingEnv::fully_monomorphized().as_query_input(ty))
        .is_ok_and(|layout| layout.is_zst())
}

fn scalar(ty: Ty<'_>) -> Option<ir::Scalar> {
    match ty.kind() {
        ty::Bool => Some(ir::Scalar::Bool),
        ty::Float(ty::FloatTy::F32) => Some(ir::Scalar::F32),
        ty::Uint(ty::UintTy::U32) => Some(ir::Scalar::U32),
        ty::Uint(ty::UintTy::U64) => Some(ir::Scalar::U64),
        ty::Uint(ty::UintTy::Usize) => Some(ir::Scalar::Usize),
        ty::Int(ty::IntTy::Isize) => Some(ir::Scalar::Isize),
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

/// Whether `ty` holds a reference or raw pointer, whose address space its uses determine. Device
/// buffer handles hold their own address space.
pub(crate) fn has_reference<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.kind() {
        ty::Ref(..) | ty::RawPtr(..) => !addresses_nothing(tcx, ty),
        ty::Pat(base, _) => has_reference(tcx, *base),
        ty::Tuple(fields) => fields.iter().any(|field| has_reference(tcx, field)),
        ty::Closure(_, arguments) => arguments
            .as_closure()
            .upvar_tys()
            .iter()
            .any(|upvar| has_reference(tcx, upvar)),
        ty::Array(element, _) | ty::Slice(element) => has_reference(tcx, *element),
        ty::Adt(definition, arguments) if device_type(tcx, definition.did()).is_none() => {
            definition.all_fields().any(|field| {
                has_reference(
                    tcx,
                    tcx.normalize_erasing_regions(
                        TypingEnv::fully_monomorphized(),
                        field.ty(tcx, arguments),
                    ),
                )
            })
        }
        _ => false,
    }
}
