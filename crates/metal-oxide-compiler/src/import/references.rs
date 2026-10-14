use metal_oxide_ir as ir;
use rustc_middle::{
    mir,
    ty::{self, Ty, TyCtxt},
};
use rustc_span::{Span, Spanned};

use super::{Result, function::FunctionImporter};
use crate::intrinsics::{BufferQuery, DeviceType, Intrinsic, SliceIterator, device_type};

/// Locals bound once to `&local` or `&mut local`, directly or by reborrow, where `local` is an
/// integer range, a slice iterator or a device buffer handle. They stand for that local in the
/// calls that use them.
pub(super) fn handles<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    normalize: impl Fn(Ty<'tcx>) -> Ty<'tcx>,
) -> Vec<Option<mir::Local>> {
    let handle = |local: mir::Local| {
        let ty = normalize(body.local_decls[local].ty);
        SliceIterator::of(tcx, ty).is_some()
            || match ty.kind() {
                ty::Adt(definition, _) => {
                    tcx.lang_items().range_struct() == Some(definition.did())
                        || device_type(tcx, definition.did())
                            .is_some_and(|kind| !matches!(kind, DeviceType::Dim3 | DeviceType::F16))
                }
                _ => false,
            }
    };
    let mut bindings = vec![0_u32; body.local_decls.len()];
    let mut handles = vec![None; body.local_decls.len()];
    for statement in body.basic_blocks.iter().flat_map(|block| &block.statements) {
        let mir::StatementKind::Assign(assignment) = &statement.kind else {
            continue;
        };
        let (destination, value) = assignment.as_ref();
        let Some(local) = destination.as_local() else {
            continue;
        };
        bindings[local.as_usize()] += 1;
        if let mir::Rvalue::Ref(_, _, referent) = value {
            handles[local.as_usize()] = match referent.as_ref() {
                mir::PlaceRef {
                    local,
                    projection: [],
                } if handle(local) => Some(local),
                mir::PlaceRef {
                    local,
                    projection: [mir::ProjectionElem::Deref],
                } => handles[local.as_usize()],
                _ => None,
            };
        }
    }
    for (handle, count) in handles.iter_mut().zip(bindings) {
        if count != 1 {
            *handle = None;
        }
    }
    handles
}

impl<'tcx> FunctionImporter<'_, 'tcx> {
    /// Types the locals that hold references. A reference points into the address space of the
    /// place it borrows; parameters and callee results carry theirs.
    pub(super) fn infer_references(
        &mut self,
        body: &mir::Body<'tcx>,
        mut pending: Vec<bool>,
    ) -> Result<()> {
        let inferred = pending.clone();
        loop {
            let mut progress = false;
            for (local, span, ty) in self.definitions(body, &inferred, &pending)? {
                if pending[local] {
                    self.locals[local] = self.checked_reference(body, local, ty, span)?;
                    pending[local] = false;
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }
        if let Some(local) = pending.iter().position(|&pending| pending) {
            return Err((
                body.local_decls[mir::Local::from_usize(local)]
                    .source_info
                    .span,
                "cannot infer the address space of this reference".into(),
            ));
        }
        for (local, span, ty) in self.definitions(body, &inferred, &pending)? {
            if self.locals[local] != ty {
                return Err((span, "a reference must point into one address space".into()));
            }
        }
        Ok(())
    }

    /// The type each assignment or call gives an `inferred` local, where it is known.
    fn definitions(
        &mut self,
        body: &mir::Body<'tcx>,
        inferred: &[bool],
        pending: &[bool],
    ) -> Result<Vec<(usize, Span, ir::Type)>> {
        let mut definitions = Vec::new();
        for block in body.basic_blocks.iter() {
            for statement in &block.statements {
                let mir::StatementKind::Assign(assignment) = &statement.kind else {
                    continue;
                };
                let (destination, value) = assignment.as_ref();
                let span = statement.source_info.span;
                if let Some(local) = destination.as_local()
                    && inferred[local.as_usize()]
                    && let Some(ty) = self.reference_value(value, pending, span)?
                {
                    definitions.push((local.as_usize(), span, ty));
                }
            }
            let terminator = block.terminator();
            if let mir::TerminatorKind::Call {
                func,
                args,
                destination,
                fn_span,
                ..
            } = &terminator.kind
                && let Some(local) = destination.as_local()
                && inferred[local.as_usize()]
                && let Some(ty) = self.call_result(
                    func,
                    args,
                    body,
                    pending,
                    terminator.source_info.span,
                    *fn_span,
                )?
            {
                definitions.push((local.as_usize(), terminator.source_info.span, ty));
            }
        }
        Ok(definitions)
    }

    /// The place's type, unless its local's type is still pending.
    fn known_type(
        &self,
        place: &mir::Place<'tcx>,
        pending: &[bool],
        span: Span,
    ) -> Result<Option<ir::Type>> {
        if !self.known(place, pending) {
            return Ok(None);
        }
        Ok(Some(self.typed_place(place, span)?.1))
    }

    /// Whether the place's local has a type; `pending` is empty once inference is done.
    fn known(&self, place: &mir::Place<'tcx>, pending: &[bool]) -> bool {
        !pending
            .get(place.local.as_usize())
            .copied()
            .unwrap_or(false)
    }

    fn reference_value(
        &mut self,
        value: &mir::Rvalue<'tcx>,
        pending: &[bool],
        span: Span,
    ) -> Result<Option<ir::Type>> {
        let (place, mutable) = match value {
            mir::Rvalue::Ref(_, mir::BorrowKind::Shared, place) => (place, false),
            mir::Rvalue::Ref(_, mir::BorrowKind::Mut { .. }, place) => (place, true),
            mir::Rvalue::RawPtr(mir::RawPtrKind::FakeForPtrMetadata, place) => (place, false),
            mir::Rvalue::Use(mir::Operand::Copy(place) | mir::Operand::Move(place), _)
            | mir::Rvalue::CopyForDeref(place) => return self.known_type(place, pending, span),
            mir::Rvalue::Cast(
                mir::CastKind::PointerCoercion(ty::adjustment::PointerCoercion::Unsize, _),
                mir::Operand::Copy(place) | mir::Operand::Move(place),
                _,
            ) => {
                if !self.known(place, pending) {
                    return Ok(None);
                }
                let (_, array) = self.typed_place(place, span)?;
                return self.array_slice(array, span).map(Some);
            }
            mir::Rvalue::Use(mir::Operand::Constant(_), _) => {
                return Err((
                    span,
                    "a reference to a constant is unsupported; borrow a local copy".into(),
                ));
            }
            mir::Rvalue::Aggregate(kind, operands) => {
                let mut components = Vec::with_capacity(operands.len());
                for operand in operands {
                    match self.argument_type(operand, pending, span)? {
                        Some(ty) => components.push(ty),
                        None => return Ok(None),
                    }
                }
                return self.aggregate_type(kind, components, span);
            }
            _ => return Err((span, "unsupported reference value".into())),
        };
        if !self.known(place, pending) {
            return Ok(None);
        }
        let place = self.located_place(place, span)?;
        let (target, slice) = match place.located {
            ir::Located::Value(ty) => (ty, false),
            ir::Located::Elements(elements) => (elements.target, true),
        };
        Ok(Some(self.module.types.intern_pointer(ir::Pointer {
            target,
            space: place.space,
            mutable,
            slice,
        })))
    }

    /// The type of an aggregate built from `components`, whose references keep their spaces.
    fn aggregate_type(
        &mut self,
        kind: &mir::AggregateKind<'tcx>,
        components: Vec<ir::Type>,
        span: Span,
    ) -> Result<Option<ir::Type>> {
        let aggregate = match kind {
            mir::AggregateKind::Tuple => ir::Aggregate::Tuple(components),
            mir::AggregateKind::Array(_) => {
                let element = components[0];
                if components.iter().any(|component| *component != element) {
                    return Err((
                        span,
                        "the references of an array must point into one address space".into(),
                    ));
                }
                ir::Aggregate::Array {
                    element,
                    length: components.len() as u32,
                }
            }
            mir::AggregateKind::Adt(definition, variant, arguments, ..)
                if self.module.tcx.adt_def(*definition).is_enum() =>
            {
                return self.variant_type(*definition, *variant, arguments, components, span);
            }
            mir::AggregateKind::Adt(definition, ..) => {
                let definition = self.module.tcx.adt_def(*definition);
                ir::Aggregate::Record {
                    name: self.module.tcx.item_name(definition.did()).to_string(),
                    fields: definition
                        .non_enum_variant()
                        .fields
                        .iter()
                        .zip(components)
                        .map(|(field, ty)| ir::RecordField {
                            name: field.name.to_string(),
                            ty,
                        })
                        .collect(),
                }
            }
            _ => return Err((span, "unsupported reference value".into())),
        };
        Ok(Some(self.module.types.intern(aggregate)))
    }

    /// The enum type of a variant built from `components`, or `None` when another variant's
    /// references leave spaces that this value does not show.
    fn variant_type(
        &mut self,
        definition: rustc_span::def_id::DefId,
        built: rustc_abi::VariantIdx,
        arguments: ty::GenericArgsRef<'tcx>,
        components: Vec<ir::Type>,
        span: Span,
    ) -> Result<Option<ir::Type>> {
        let tcx = self.module.tcx;
        let enumeration = tcx.adt_def(definition);
        let ty::Adt(_, arguments) = *self
            .normalize_type(ty::Ty::new_adt(tcx, enumeration, arguments))
            .kind()
        else {
            unreachable!("an enum type stays an enum when normalized")
        };
        let mut built_fields = Some(components);
        let mut variants = Vec::with_capacity(enumeration.variants().len());
        for (index, variant) in enumeration.variants().iter_enumerated() {
            if index == built {
                variants.push(built_fields.take().unwrap());
                continue;
            }
            let mut fields = Vec::with_capacity(variant.fields.len());
            for field in &variant.fields {
                let ty = tcx.normalize_erasing_regions(
                    ty::TypingEnv::fully_monomorphized(),
                    field.ty(tcx, arguments),
                );
                if crate::types::has_reference(tcx, ty) {
                    return Ok(None);
                }
                fields.push(self.lower_type(ty, span)?);
            }
            variants.push(fields);
        }
        Ok(Some(self.module.types.intern(ir::Aggregate::enumeration(
            tcx.item_name(definition).to_string(),
            variants,
        ))))
    }

    /// The slice an array reference coerces to.
    pub(super) fn array_slice(&mut self, array: ir::Type, span: Span) -> Result<ir::Type> {
        let table = self.module.types.table();
        let pointer = table
            .pointer(array)
            .ok_or((span, "unsized coercions require an array reference".into()))?;
        let ir::Type::Aggregate(id) = pointer.target else {
            return Err((span, "unsized coercions require an array reference".into()));
        };
        let Some(ir::Aggregate::Array { element, .. }) = table.get(id) else {
            return Err((span, "unsized coercions require an array reference".into()));
        };
        let slice = ir::Pointer {
            target: *element,
            slice: true,
            ..pointer
        };
        Ok(self.module.types.intern_pointer(slice))
    }

    fn call_result(
        &mut self,
        func: &mir::Operand<'tcx>,
        args: &[Spanned<mir::Operand<'tcx>>],
        body: &mir::Body<'tcx>,
        pending: &[bool],
        span: Span,
        fn_span: Span,
    ) -> Result<Option<ir::Type>> {
        let callee = self.callee(func, body, span)?;
        if let Some(iterator) =
            crate::intrinsics::slice_next(self.module.tcx, callee.definition, callee.arguments)
        {
            return self.next_item_type(iterator, &args[0].node, pending, span);
        }
        if let Some(Intrinsic::SliceIterator(iterator)) = callee.intrinsic {
            let Some(slice) = self.argument_type(&args[0].node, pending, args[0].span)? else {
                return Ok(None);
            };
            return self.iterator_type(iterator, slice, span).map(Some);
        }
        if let Some(Intrinsic::Buffer(query @ (BufferQuery::Slice | BufferQuery::Element))) =
            callee.intrinsic
        {
            let parameter = self.buffer_parameter(&args[0].node, span)?;
            let slice = self.buffer_slice_type(parameter, span)?;
            if query == BufferQuery::Slice {
                return Ok(Some(slice));
            }
            let element = self.module.types.table().pointer(slice).unwrap();
            let reference = self.module.types.intern_pointer(ir::Pointer {
                slice: false,
                ..element
            });
            return Ok(Some(self.module.types.option(reference)));
        }
        let Some(super::calls::Signature { types, messages }) =
            self.signature(args, body, pending)?
        else {
            return Ok(None);
        };
        match callee.intrinsic {
            Some(Intrinsic::Identity) => Ok(types.first().copied()),
            Some(Intrinsic::Subslice) => {
                let receiver = types[0];
                let array = self
                    .module
                    .types
                    .table()
                    .pointer(receiver)
                    .is_some_and(|pointer| !pointer.slice);
                if array {
                    self.array_slice(receiver, span).map(Some)
                } else {
                    Ok(Some(receiver))
                }
            }
            Some(intrinsic) => Err((
                span,
                format!("{} does not return a reference", intrinsic.name()),
            )),
            None => {
                let caller = self.caller_of(&callee, fn_span);
                let id = self
                    .module
                    .function(callee.instance, types, messages, caller, false)?;
                Ok(Some(self.module.return_type(id)))
            }
        }
    }

    /// The type an argument passes, unless it reads a reference whose type is still pending.
    pub(super) fn argument_type(
        &mut self,
        operand: &mir::Operand<'tcx>,
        pending: &[bool],
        span: Span,
    ) -> Result<Option<ir::Type>> {
        match operand {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => {
                self.known_type(place, pending, span)
            }
            mir::Operand::Constant(constant) => {
                let ty = constant.ty();
                self.lower_type(ty, span).map(Some)
            }
            _ => Err((span, "unsupported MIR operand".into())),
        }
    }

    /// `ty` after checking that it matches the local's Rust reference type.
    fn checked_reference(
        &mut self,
        body: &mir::Body<'tcx>,
        local: usize,
        ty: ir::Type,
        span: Span,
    ) -> Result<ir::Type> {
        let rust = self.normalize_type(body.local_decls[mir::Local::from_usize(local)].ty);
        let mut spaces = Vec::new();
        self.module.types.spaces(ty, &mut spaces);
        let mut spaces = spaces.into_iter();
        if self.module.types.with_spaces(rust, &mut spaces, span)? != ty || spaces.next().is_some()
        {
            return Err((span, format!("unsupported reference type: {rust}")));
        }
        Ok(ty)
    }
}
