use metal_oxide_ir as ir;
use rustc_middle::{mir, ty::Instance};
use rustc_span::Span;

use super::{
    FunctionKey, ModuleImporter, Result, control, handles, location, messages, panics::Panics,
    ranges,
};
use crate::intrinsics::Expanded;

pub(super) struct FunctionImporter<'a, 'tcx> {
    pub(super) module: &'a mut ModuleImporter<'tcx>,
    pub(super) instance: Instance<'tcx>,
    /// The kernel's lowered signature, or the argument types of the helper's callers.
    pub(super) parameters: Vec<ir::Type>,
    /// The literal of each local that holds a message, starting from the parameters'.
    pub(super) messages: Vec<Option<String>>,
    /// Where the checks of a `#[track_caller]` function report: its outermost caller.
    pub(super) caller: Option<ir::SourceLocation>,
    /// The return type the caller gives a function whose return values hold no reference.
    returns: Option<ir::Type>,
    pub(super) kernel: bool,
    pub(super) locals: Vec<ir::Type>,
    pub(super) allocations: u32,
    pub(super) constants: Vec<Option<ir::Constant>>,
    pub(super) handles: Vec<Option<mir::Local>>,
    /// Blocks numbered after the MIR blocks, from expanded range iteration and subslicing.
    pub(super) appended: Vec<ir::Block>,
}

impl<'a, 'tcx> FunctionImporter<'a, 'tcx> {
    pub(super) fn new(module: &'a mut ModuleImporter<'tcx>, key: FunctionKey<'tcx>) -> Self {
        Self {
            module,
            instance: key.instance,
            parameters: key.parameters,
            messages: key.messages,
            caller: key.caller,
            returns: key.returns,
            kernel: false,
            locals: Vec::new(),
            allocations: 0,
            constants: Vec::new(),
            handles: Vec::new(),
            appended: Vec::new(),
        }
    }

    pub(super) fn import(mut self, kernel: bool) -> Result<ir::Function> {
        self.kernel = kernel;
        let tcx = self.module.tcx;
        let body = crate::monomorphize::body(tcx, self.instance);
        let source = location(tcx, tcx.def_span(self.instance.def_id()));
        let used = control::used_locals(body);
        self.messages = messages::messages(tcx, body, &self.messages);
        let panics = Panics::new(tcx, body, self.messages.clone())?;
        self.constants = super::constants::locals(tcx, self.instance, body);
        self.handles = handles::handles(tcx, body, |ty| self.normalize_type(ty));
        let discriminants = ranges::discriminants(body);
        let mut pending = vec![false; body.local_decls.len()];
        for (index, local) in body.local_decls.iter().enumerate() {
            let ty = if (1..=body.arg_count).contains(&index) {
                self.parameters[index - 1]
            } else if let Some(returns) = self.returns.filter(|_| index == 0) {
                returns
            } else if !used[index]
                || self.handles[index].is_some()
                || panics.is_literal(index)
                || discriminants[index].is_some()
                || messages::is_message(self.normalize_type(local.ty))
            {
                ir::Type::Unit
            } else if crate::types::has_reference(tcx, self.normalize_type(local.ty)) {
                pending[index] = true;
                ir::Type::Unit
            } else {
                self.lower_type(local.ty, local.source_info.span)?
            };
            self.locals.push(ty);
        }
        self.infer_references(body, pending)?;
        // A discriminant has its enum's tag type, known once references are.
        for (index, place) in discriminants.iter().enumerate() {
            if let Some(place) = place.filter(|_| used[index]) {
                let span = body.local_decls[mir::Local::from_usize(index)]
                    .source_info
                    .span;
                let (_, ty) = self.typed_place(&place, span)?;
                if ty == ir::Type::Scalar(ir::Scalar::I8) {
                    self.locals[index] = ty;
                    continue;
                }
                let tag = self
                    .module
                    .types
                    .tag(ty)
                    .ok_or((span, "discriminants require an enum".into()))?;
                self.locals[index] = ir::Type::Scalar(tag);
            }
        }
        let mut blocks = Vec::with_capacity(body.basic_blocks.len());
        for block in body.basic_blocks.iter() {
            blocks.push(self.block(block, body, &panics)?);
        }
        blocks.append(&mut self.appended);
        Ok(ir::Function {
            name: if kernel {
                tcx.item_name(self.instance.def_id()).to_string()
            } else {
                self.instance.to_string()
            },
            kernel,
            required_block: if kernel {
                crate::collect::block_shape(tcx, self.instance.def_id())?
            } else {
                None
            },
            requires: if kernel {
                self.requirements(body)?
            } else {
                self.invariants(body)
            },
            parameters: body.arg_count,
            locals: self.locals,
            blocks,
            source,
        })
    }

    /// Where a check at `span` reports, as Rust's `#[track_caller]` panics would: the caller of
    /// a `#[track_caller]` function, otherwise the outermost macro call that expanded to it.
    pub(super) fn check_location(&self, span: rustc_span::Span) -> ir::SourceLocation {
        self.caller.clone().unwrap_or_else(|| {
            let span = span.ctxt().outer_expn().expansion_cause().unwrap_or(span);
            location(self.module.tcx, span)
        })
    }

    fn block(
        &mut self,
        block: &mir::BasicBlockData<'tcx>,
        body: &mir::Body<'tcx>,
        panics: &Panics,
    ) -> Result<ir::Block> {
        let mut statements = Vec::new();
        for statement in &block.statements {
            let span = statement.source_info.span;
            match &statement.kind {
                // Handles, messages and uninhabited values need no device statements.
                mir::StatementKind::Assign(assignment)
                    if assignment.0.as_local().is_some_and(|local| {
                        self.handles[local.as_usize()].is_some()
                            || messages::is_message(self.normalize_type(body.local_decls[local].ty))
                            || self.locals[local.as_usize()] == ir::Type::Never
                    }) => {}
                mir::StatementKind::Assign(assignment) => {
                    let (destination, value) = assignment.as_ref();
                    let (destination, ty) = self.typed_place(destination, span)?;
                    // Zero-sized values, and references to them, hold nothing to assign.
                    if ty == ir::Type::Unit {
                        continue;
                    }
                    if let mir::Rvalue::Cast(mir::CastKind::Transmute, operand, to) = value
                        && self.module.types.tag(ty).is_some()
                    {
                        let niche = self.niche_transmute(destination, ty, operand, *to, span)?;
                        statements.extend(niche);
                        continue;
                    }
                    statements.push(ir::Statement {
                        destination,
                        value: self.expression(value, ty, span)?,
                        source: location(self.module.tcx, span),
                    });
                }
                mir::StatementKind::SetDiscriminant {
                    place,
                    variant_index,
                } => {
                    let (mut destination, ty) = self.typed_place(place, span)?;
                    let tag = self
                        .module
                        .types
                        .tag(ty)
                        .and_then(|tag| control::integer(tag, variant_index.as_u32().into()))
                        .ok_or((span, "discriminants require an enum".into()))?;
                    destination.projection.push(ir::Projection::Field(0));
                    statements.push(ir::Statement {
                        destination,
                        value: ir::Expression::Use(ir::Operand::Constant(tag)),
                        source: location(self.module.tcx, span),
                    });
                }
                mir::StatementKind::StorageLive(_)
                | mir::StatementKind::StorageDead(_)
                | mir::StatementKind::Nop => {}
                mir::StatementKind::Intrinsic(intrinsic)
                    if matches!(**intrinsic, mir::NonDivergingIntrinsic::Assume(_)) => {}
                other => return Err((span, format!("unsupported MIR statement: {other:?}"))),
            }
        }
        let term = block.terminator();
        let span = term.source_info.span;
        let reported = self.check_location(span);
        let mut source = location(self.module.tcx, span);
        if let Some(message) = panics.message(self.module.tcx, &term.kind) {
            return Ok(ir::Block {
                statements,
                terminator: ir::Terminator::Panic(message),
                source: reported,
            });
        }
        let terminator = match &term.kind {
            mir::TerminatorKind::Goto { target } => ir::Terminator::Goto(target.as_usize()),
            mir::TerminatorKind::Return => ir::Terminator::Return,
            mir::TerminatorKind::Unreachable => ir::Terminator::Unreachable,
            mir::TerminatorKind::SwitchInt { discr, targets } => {
                self.switch(discr, targets, body, span)?
            }
            mir::TerminatorKind::Assert {
                cond,
                expected,
                msg,
                target,
                ..
            } => {
                source = reported;
                ir::Terminator::Assert {
                    condition: self.operand(cond, span)?,
                    expected: *expected,
                    enabled: (self.module.tcx.sess.overflow_checks()
                        || !msg.is_optional_overflow_check())
                        && self.operand(cond, span)?
                            != ir::Operand::Constant(ir::Constant::Bool(*expected)),
                    target: target.as_usize(),
                    check: check(msg, span)?,
                }
            }
            mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target: Some(target),
                ..
            } if self.expanded_call(func, args, body, span)?.is_some() => {
                let (expanded, place) = self.expanded_call(func, args, body, span)?.unwrap();
                let target = target.as_usize();
                let local = place.local;
                match expanded {
                    Expanded::Range => {
                        self.range_next(&mut statements, local, destination, target, body, span)?
                    }
                    Expanded::Chunks => {
                        self.next_chunk(&mut statements, local, destination, target, body, span)?
                    }
                    Expanded::Array => {
                        self.array_next(&mut statements, place, destination, target, body, span)?
                    }
                }
            }
            mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target: Some(target),
                ..
            } if self.is_subslice(func, body, span)? => self.subslice(
                &mut statements,
                args,
                destination,
                target.as_usize(),
                body,
                span,
            )?,
            mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target: Some(target),
                ..
            } if self.buffer_query(func, body, span)?.is_some() => {
                let query = self.buffer_query(func, body, span)?.unwrap();
                self.buffer(
                    &mut statements,
                    query,
                    args,
                    destination,
                    target.as_usize(),
                    body,
                    span,
                )?
            }
            mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target: Some(target),
                ..
            } if matches!(
                self.callee(func, body, span)?.intrinsic,
                Some(crate::intrinsics::Intrinsic::Vector(_))
            ) =>
            {
                let Some(crate::intrinsics::Intrinsic::Vector(op)) =
                    self.callee(func, body, span)?.intrinsic
                else {
                    unreachable!()
                };
                self.vector_call(
                    op,
                    args,
                    (destination, target.as_usize()),
                    body,
                    &mut statements,
                    span,
                )?
            }
            mir::TerminatorKind::Call {
                func,
                args,
                target: Some(target),
                ..
            } if matches!(
                self.callee(func, body, span)?.intrinsic,
                Some(crate::intrinsics::Intrinsic::Swap)
            ) =>
            {
                self.swap(&mut statements, args, span)?;
                ir::Terminator::Goto(target.as_usize())
            }
            mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target: Some(target),
                ..
            } if matches!(
                self.callee(func, body, span)?.intrinsic,
                Some(crate::intrinsics::Intrinsic::ArrayFromClosure(_))
            ) =>
            {
                let callee = self.callee(func, body, span)?;
                let Some(crate::intrinsics::Intrinsic::ArrayFromClosure(mapped)) = callee.intrinsic
                else {
                    unreachable!()
                };
                let call = (callee.instance, mapped);
                let result = (destination, target.as_usize());
                self.array_from_closure(&mut statements, args, call, result, body, span)?
            }
            mir::TerminatorKind::Call {
                func,
                args,
                target: Some(target),
                ..
            } if matches!(
                self.callee(func, body, span)?.intrinsic,
                Some(crate::intrinsics::Intrinsic::CopyNonoverlapping)
            ) =>
            {
                self.copy_nonoverlapping(&mut statements, args, target.as_usize(), body, span)?
            }
            mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target: Some(target),
                fn_span,
                ..
            } => {
                let (destination, ty) = self.typed_place(destination, span)?;
                statements.push(ir::Statement {
                    destination,
                    value: self.call(func, args, body, ty, span, *fn_span)?,
                    source: source.clone(),
                });
                ir::Terminator::Goto(target.as_usize())
            }
            mir::TerminatorKind::Drop { place, target, .. } => {
                let ty = self.normalize_type(place.ty(&body.local_decls, self.module.tcx).ty);
                if crate::intrinsics::needs_drop(self.module.tcx, ty) {
                    return Err((span, "destructors cannot be imported".into()));
                }
                ir::Terminator::Goto(target.as_usize())
            }
            other => return Err((span, format!("unsupported MIR terminator: {other:?}"))),
        };
        Ok(ir::Block {
            statements,
            terminator,
            source,
        })
    }
}

fn check(message: &mir::AssertMessage<'_>, span: Span) -> Result<ir::Check> {
    use mir::{AssertKind, BinOp};
    Ok(match message {
        AssertKind::BoundsCheck { .. } => ir::Check::Bounds,
        AssertKind::Overflow(op, ..) => ir::Check::Overflow(match op {
            BinOp::Add => ir::OverflowOp::Add,
            BinOp::Sub => ir::OverflowOp::Sub,
            BinOp::Mul => ir::OverflowOp::Mul,
            BinOp::Div => ir::OverflowOp::Div,
            BinOp::Rem => ir::OverflowOp::Rem,
            BinOp::Shl => ir::OverflowOp::Shl,
            BinOp::Shr => ir::OverflowOp::Shr,
            other => return Err((span, format!("unsupported overflow assertion: {other:?}"))),
        }),
        AssertKind::OverflowNeg(_) => ir::Check::Overflow(ir::OverflowOp::Neg),
        AssertKind::DivisionByZero(_) => ir::Check::DivisionByZero,
        AssertKind::RemainderByZero(_) => ir::Check::RemainderByZero,
        other => return Err((span, format!("unsupported MIR assertion: {other:?}"))),
    })
}

impl<'tcx> FunctionImporter<'_, 'tcx> {
    /// The iterator and its place when a call is a `next` that the importer expands.
    fn expanded_call(
        &self,
        func: &mir::Operand<'tcx>,
        args: &[rustc_span::Spanned<mir::Operand<'tcx>>],
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<Option<(Expanded, ir::Place)>> {
        let callee = self.callee(func, body, span)?;
        Ok(self.expanded_next(&callee, args))
    }

    pub(super) fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    /// Appends a block after the MIR blocks and returns its id.
    pub(super) fn append(&mut self, body: &mir::Body<'_>, block: ir::Block) -> usize {
        self.appended.push(block);
        body.basic_blocks.len() + self.appended.len() - 1
    }
}
