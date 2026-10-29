use metal_oxide_ir as ir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind};
use rustc_middle::{
    mir,
    ty::{self, Instance, TyCtxt, consts::ConstExt},
};
use rustc_span::{Span, Spanned};

use super::{FunctionKey, Result, function::FunctionImporter};
use crate::intrinsics::Intrinsic;

/// Whether every value the instance returns is built from values without references, such as
/// `None` or `ControlFlow::Continue(())`, so its return type's references point where its
/// caller's do.
pub(super) fn returns_no_references<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> bool {
    let body = crate::monomorphize::body(tcx, instance);
    let referenceless = |operand: &mir::Operand<'tcx>| {
        let ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            ty::TypingEnv::fully_monomorphized(),
            ty::EarlyBinder::bind(tcx, operand.ty(&body.local_decls, tcx)),
        );
        !crate::types::has_reference(tcx, ty)
    };
    let mut returned = false;
    for block in body.basic_blocks.iter() {
        if let mir::TerminatorKind::Call { destination, .. } = &block.terminator().kind
            && destination.local == mir::RETURN_PLACE
        {
            return false;
        }
        for statement in &block.statements {
            match statement.kind.as_assign() {
                Some((place, mir::Rvalue::Aggregate(kind, operands)))
                    if *place == mir::Place::return_place()
                        && operands.iter().all(referenceless)
                        && matches!(**kind, mir::AggregateKind::Adt(..)) =>
                {
                    returned = true;
                }
                Some((place, _)) if place.local == mir::RETURN_PLACE => return false,
                _ => {}
            }
        }
    }
    returned
}

/// What a call passes: argument types, and for each message argument its literal, which
/// specializes the callee in place of a value.
pub(super) struct Signature {
    pub types: Vec<ir::Type>,
    pub messages: Vec<Option<String>>,
}

/// A resolved call target.
pub(super) struct Callee<'tcx> {
    pub instance: Instance<'tcx>,
    pub definition: rustc_hir::def_id::DefId,
    /// The generic arguments of the called definition, before resolution.
    pub arguments: ty::GenericArgsRef<'tcx>,
    pub intrinsic: Option<Intrinsic>,
}

impl<'tcx> FunctionImporter<'_, 'tcx> {
    pub(super) fn callee(
        &self,
        func: &mir::Operand<'tcx>,
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<Callee<'tcx>> {
        let tcx = self.module.tcx;
        let ty = self.normalize_type(func.ty(&body.local_decls, tcx));
        let ty::FnDef(definition, arguments) = *ty.kind() else {
            return Err((span, "indirect calls cannot be imported".into()));
        };
        let arguments = tcx.instantiate_bound_regions_with_erased(arguments);
        let instance = Instance::try_resolve(
            tcx,
            ty::TypingEnv::fully_monomorphized(),
            definition,
            arguments,
        )
        .ok()
        .flatten()
        .ok_or((span, "cannot resolve concrete function".into()))?;
        let intrinsic = crate::intrinsics::builtin(tcx, definition)
            .or_else(|| crate::intrinsics::builtin(tcx, instance.def_id()));
        Ok(Callee {
            instance,
            definition,
            arguments,
            intrinsic,
        })
    }

    /// A call at `span`; `fn_span` covers its function and arguments, the location Rust's
    /// `#[track_caller]` reports.
    pub(super) fn call(
        &mut self,
        func: &mir::Operand<'tcx>,
        args: &[Spanned<mir::Operand<'tcx>>],
        body: &mir::Body<'tcx>,
        destination: ir::Type,
        span: Span,
        fn_span: Span,
    ) -> Result<ir::Expression> {
        let callee = self.callee(func, body, span)?;
        if let DefKind::Ctor(of, CtorKind::Fn) = self.module.tcx.def_kind(callee.definition) {
            return self.construct(&callee, of, args, destination, span);
        }
        if let Some(Intrinsic::Log) = callee.intrinsic {
            return self.log(args, callee.arguments, span);
        }
        if let Some(Intrinsic::Atomic(op)) = callee.intrinsic {
            return self.atomic(op, args, body, span);
        }
        if let Some(intrinsic) = callee.intrinsic {
            let arguments = args
                .iter()
                .map(|a| self.operand(&a.node, a.span))
                .collect::<Result<Vec<_>>>()?;
            return self.intrinsic(intrinsic, callee.arguments, arguments, span);
        }
        // A call would import the kernel as a helper before its entry does.
        if self.module.entries.contains(&callee.instance) {
            return Err((span, "calling a kernel entrypoint is unsupported".into()));
        }
        let args = self.spread(func, args, body)?;
        let args = args.as_slice();
        let Signature { types, messages } = self
            .signature(callee.instance, args, body, &[])?
            .expect("inferred references are known");
        let arguments = args
            .iter()
            .zip(&types)
            .map(|(argument, ty)| match ty {
                ir::Type::Unit => Ok(ir::Operand::Constant(ir::Constant::Unit)),
                _ => self.operand(&argument.node, argument.span),
            })
            .collect::<Result<Vec<_>>>()?;
        let key = FunctionKey {
            instance: callee.instance,
            entry: None,
            parameters: types,
            messages,
            caller: self.caller_of(&callee, fn_span),
            returns: (!self.module.types.table().is_owned(destination)
                && returns_no_references(self.module.tcx, callee.instance))
            .then_some(destination),
        };
        Ok(ir::Expression::Call {
            function: self.module.function(key, false)?,
            arguments,
        })
    }

    /// A tuple struct or variant constructor called as a function, as `Err` in
    /// `map_or(Ok(()), Err)`, builds its value like the aggregate it stands for.
    fn construct(
        &mut self,
        callee: &Callee<'tcx>,
        of: CtorOf,
        args: &[Spanned<mir::Operand<'tcx>>],
        destination: ir::Type,
        span: Span,
    ) -> Result<ir::Expression> {
        let tcx = self.module.tcx;
        let definition = match of {
            CtorOf::Struct => tcx.parent(callee.definition),
            CtorOf::Variant => tcx.parent(tcx.parent(callee.definition)),
        };
        let adt = tcx.adt_def(definition);
        let variant = adt.variant_index_with_ctor_id(callee.definition);
        let kind = mir::AggregateKind::Adt(adt.did(), variant, callee.arguments, None, None);
        let fields = args.iter().map(|argument| argument.node.clone());
        self.expression(
            &mir::Rvalue::Aggregate(Box::new(kind), fields.collect()),
            destination,
            span,
        )
    }

    /// The arguments a call passes to its callee's parameters. A `rust-call` call to a closure
    /// passes the fields of its last argument's tuple as separate parameters.
    pub(super) fn spread(
        &self,
        func: &mir::Operand<'tcx>,
        args: &[Spanned<mir::Operand<'tcx>>],
        body: &mir::Body<'tcx>,
    ) -> Result<Vec<Spanned<mir::Operand<'tcx>>>> {
        let tcx = self.module.tcx;
        let ty::FnDef(definition, _) = *self.normalize_type(func.ty(&body.local_decls, tcx)).kind()
        else {
            return Ok(args.to_vec());
        };
        let Some((last, rest)) = args.split_last().filter(|_| {
            tcx.fn_sig(definition).skip_binder().abi() == rustc_abi::ExternAbi::RustCall
        }) else {
            return Ok(args.to_vec());
        };
        // A closure body takes its arguments one by one; a call shim takes the tuple.
        let callee = self.callee(func, body, last.span)?;
        if crate::monomorphize::body(tcx, callee.instance)
            .spread_arg
            .is_some()
        {
            return Ok(args.to_vec());
        }
        let (place, moved) = match last.node {
            mir::Operand::Copy(place) => (place, false),
            mir::Operand::Move(place) => (place, true),
            _ => return Err((last.span, "closure arguments must be in a local".into())),
        };
        let ty::Tuple(fields) = *self
            .normalize_type(place.ty(&body.local_decls, tcx).ty)
            .kind()
        else {
            return Err((last.span, "closure arguments must be a tuple".into()));
        };
        let mut spread = rest.to_vec();
        for (index, field) in fields.iter().enumerate() {
            let place = tcx.mk_place_field(place, rustc_abi::FieldIdx::from_usize(index), field);
            spread.push(Spanned {
                node: if moved {
                    mir::Operand::Move(place)
                } else {
                    mir::Operand::Copy(place)
                },
                span: last.span,
            });
        }
        Ok(spread)
    }

    /// The location a `#[track_caller]` callee's checks report: this function's own caller
    /// when it tracks one too, otherwise the call's `fn_span`.
    pub(super) fn caller_of(
        &self,
        callee: &Callee<'tcx>,
        fn_span: Span,
    ) -> Option<ir::SourceLocation> {
        callee
            .instance
            .def
            .requires_caller_location(self.module.tcx)
            .then(|| self.check_location(fn_span))
    }

    /// What a call passes, unless an argument reads a reference whose type is still
    /// `pending`.
    pub(super) fn signature(
        &mut self,
        callee: Instance<'tcx>,
        args: &[Spanned<mir::Operand<'tcx>>],
        body: &mir::Body<'tcx>,
        pending: &[bool],
    ) -> Result<Option<Signature>> {
        let tcx = self.module.tcx;
        let parameters = &crate::monomorphize::body(tcx, callee).local_decls;
        let mut types = Vec::with_capacity(args.len());
        let mut messages = Vec::with_capacity(args.len());
        for (index, argument) in args.iter().enumerate() {
            let rust = self.normalize_type(argument.node.ty(&body.local_decls, tcx));
            if super::messages::is_message(rust) {
                types.push(ir::Type::Unit);
                messages.push(super::messages::message(
                    tcx,
                    &argument.node,
                    &self.messages,
                ));
                continue;
            }
            let Some(ty) = self.argument_type(&argument.node, pending, argument.span)? else {
                return Ok(None);
            };
            // A shared reference parameter, such as a `Fn` closure's environment called through
            // `FnMut`, takes a mutable argument as a reborrow.
            let shared = matches!(
                callee
                    .instantiate_mir_and_normalize_erasing_regions(
                        tcx,
                        ty::TypingEnv::fully_monomorphized(),
                        ty::EarlyBinder::bind(
                            tcx,
                            parameters[mir::Local::from_usize(index + 1)].ty
                        ),
                    )
                    .kind(),
                ty::Ref(_, _, ty::Mutability::Not)
            );
            types.push(match self.module.types.table().pointer(ty) {
                Some(pointer) if shared && pointer.mutable && !pointer.slice => {
                    self.module.types.intern_pointer(ir::Pointer {
                        mutable: false,
                        ..pointer
                    })
                }
                _ => ty,
            });
            messages.push(None);
        }
        Ok(Some(Signature { types, messages }))
    }

    /// `log::write(format, tuple)`: the literal format and one operand per tuple field.
    fn log(
        &mut self,
        args: &[Spanned<mir::Operand<'tcx>>],
        types: ty::GenericArgsRef<'tcx>,
        span: Span,
    ) -> Result<ir::Expression> {
        let format = super::panics::literal(self.module.tcx, &args[0].node)
            .ok_or((span, "device log formats must be string literals".into()))?;
        let ty::Tuple(fields) = types.type_at(0).kind() else {
            return Err((span, "device log arguments must be a tuple".into()));
        };
        let arguments = match self.operand(&args[1].node, args[1].span)? {
            ir::Operand::Place(place) => (0..fields.len() as u32)
                .map(|field| {
                    let mut place = place.clone();
                    place.projection.push(ir::Projection::Field(field));
                    ir::Operand::Place(place)
                })
                .collect(),
            ir::Operand::AggregateConstant { fields, .. } => fields,
            ir::Operand::Constant(ir::Constant::Unit) => Vec::new(),
            ir::Operand::Constant(_) => {
                return Err((span, "device log arguments must be a tuple".into()));
            }
        };
        Ok(ir::Expression::Log { format, arguments })
    }

    fn intrinsic(
        &mut self,
        intrinsic: Intrinsic,
        types: ty::GenericArgsRef<'tcx>,
        values: Vec<ir::Operand>,
        span: Span,
    ) -> Result<ir::Expression> {
        Ok(match intrinsic {
            Intrinsic::Math(op) => {
                if values.len() != op.arity() {
                    return Err((span, "invalid math intrinsic arity".into()));
                }
                ir::Expression::Math {
                    op,
                    arguments: values,
                }
            }
            Intrinsic::Integer(op) => {
                if values.len() != op.arity() {
                    return Err((span, "invalid integer intrinsic arity".into()));
                }
                ir::Expression::Integer {
                    op,
                    arguments: values,
                }
            }
            Intrinsic::Bitcast(to) => {
                let [value] = arguments(values, span)?;
                ir::Expression::Bitcast(value, to)
            }
            Intrinsic::FloatConvert { to, .. } => {
                let [value] = arguments(values, span)?;
                ir::Expression::Cast(value, to)
            }
            Intrinsic::FloatBinary(_, op) => {
                let [left, right] = arguments(values, span)?;
                ir::Expression::Binary(op, left, right)
            }
            Intrinsic::Vector(_) => unreachable!("vectors are scalarized in place"),
            Intrinsic::Log => unreachable!("log calls are imported before their operands"),
            Intrinsic::UncheckedFloatToInt => {
                let [value] = arguments(values, span)?;
                let ir::Type::Scalar(to) = self.lower_type(types.type_at(0), span)? else {
                    return Err((span, "unchecked float conversion requires a scalar".into()));
                };
                ir::Expression::UncheckedCast(value, to)
            }
            Intrinsic::WrappingShift(op) => {
                let [value, count] = arguments(values, span)?;
                ir::Expression::Binary(op, value, count)
            }
            Intrinsic::Coordinates(builtin) => {
                arguments::<0>(values, span)?;
                ir::Expression::Coordinates(builtin)
            }
            Intrinsic::SimdCoordinate(builtin) => {
                arguments::<0>(values, span)?;
                ir::Expression::SimdCoordinate(builtin)
            }
            // `ThreadIndex::get` passes its witness, which holds nothing.
            Intrinsic::ThreadIndex if values.len() > 1 => {
                return Err((span, "invalid device builtin arguments".into()));
            }
            Intrinsic::ThreadIndex => ir::Expression::ThreadIndex,
            Intrinsic::Simd(op) => {
                if values.len() != op.arity() {
                    return Err((span, "invalid SIMD intrinsic arity".into()));
                }
                ir::Expression::Simd {
                    op,
                    arguments: values,
                }
            }
            Intrinsic::Matrix(op) => {
                if values.len() != op.arity() {
                    return Err((span, "invalid device builtin arguments".into()));
                }
                ir::Expression::Matrix {
                    op,
                    arguments: values,
                }
            }
            Intrinsic::ThreadgroupAlloc(access) => {
                arguments::<0>(values, span)?;
                let ir::Type::Scalar(element) = self.lower_type(types.type_at(0), span)? else {
                    return Err((span, "threadgroup elements must be scalars".into()));
                };
                let length = types
                    .const_at(1)
                    .try_to_target_usize(self.module.tcx)
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or((
                        span,
                        "threadgroup length must be a concrete u32-sized constant".into(),
                    ))?;
                let id = self.allocations;
                self.allocations += 1;
                ir::Expression::ThreadgroupAlloc {
                    id,
                    element,
                    length,
                    access,
                }
            }
            Intrinsic::ThreadgroupBarrier => {
                arguments::<0>(values, span)?;
                ir::Expression::ThreadgroupBarrier
            }
            Intrinsic::Uninit => {
                arguments::<0>(values, span)?;
                let array = ty::Ty::new_array_with_const_len(
                    self.module.tcx,
                    types.type_at(0),
                    types.const_at(1),
                );
                ir::Expression::Uninit(self.lower_type(array, span)?)
            }
            Intrinsic::Atomic(_) => {
                unreachable!("atomics are imported before their ordering arguments")
            }
            Intrinsic::BufferLoad => {
                let [buffer, index] = arguments(values, span)?;
                ir::Expression::BufferLoad { buffer, index }
            }
            Intrinsic::BufferStore => {
                let [buffer, index, value] = arguments(values, span)?;
                ir::Expression::BufferStore {
                    buffer,
                    index,
                    value,
                }
            }
            Intrinsic::SliceLen => {
                let [slice] = arguments(values, span)?;
                ir::Expression::Len(slice)
            }
            Intrinsic::Identity => {
                let [value] = arguments(values, span)?;
                ir::Expression::Use(value)
            }
            Intrinsic::StaticallyKnown => {
                arguments::<1>(values, span)?;
                ir::Expression::Use(ir::Operand::Constant(ir::Constant::Bool(false)))
            }
            Intrinsic::OffsetFrom => {
                let [end, start] = arguments(values, span)?;
                ir::Expression::Binary(ir::BinaryOp::OffsetFrom, end, start)
            }
            Intrinsic::Subslice | Intrinsic::Buffer(_) => {
                unreachable!("subslices and buffer queries are expanded in place")
            }
        })
    }
}

fn arguments<const N: usize>(values: Vec<ir::Operand>, span: Span) -> Result<[ir::Operand; N]> {
    values
        .try_into()
        .map_err(|_| (span, "invalid device builtin arguments".into()))
}
