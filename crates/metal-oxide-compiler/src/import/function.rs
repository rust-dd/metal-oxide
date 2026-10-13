use metal_oxide_ir as ir;
use rustc_middle::{mir, ty::Instance};
use rustc_span::Span;

use super::{ModuleImporter, Result, control, location, panics::Panics, ranges, references};

pub(super) struct FunctionImporter<'a, 'tcx> {
    pub(super) module: &'a mut ModuleImporter<'tcx>,
    pub(super) instance: Instance<'tcx>,
    /// The kernel's lowered signature, or the argument types of the helper's callers.
    parameters: Vec<ir::Type>,
    pub(super) kernel: bool,
    pub(super) locals: Vec<ir::Type>,
    pub(super) allocations: u32,
    pub(super) constants: Vec<Option<ir::Constant>>,
    pub(super) handles: Vec<Option<mir::Local>>,
    /// Blocks numbered after the MIR blocks, from expanded range iteration and subslicing.
    pub(super) appended: Vec<ir::Block>,
}

impl<'a, 'tcx> FunctionImporter<'a, 'tcx> {
    pub(super) fn new(
        module: &'a mut ModuleImporter<'tcx>,
        instance: Instance<'tcx>,
        parameters: Vec<ir::Type>,
    ) -> Self {
        Self {
            module,
            instance,
            parameters,
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
        let body = tcx.instance_mir(self.instance.def);
        let source = location(tcx, tcx.def_span(self.instance.def_id()));
        let used = control::used_locals(body);
        let panics = Panics::new(tcx, body)?;
        self.constants = super::constants::locals(tcx, self.instance, body);
        self.handles = references::handles(tcx, body, |ty| self.normalize_type(ty));
        let discriminants = ranges::discriminants(body);
        let mut pending = vec![false; body.local_decls.len()];
        for (index, local) in body.local_decls.iter().enumerate() {
            let ty = if (1..=body.arg_count).contains(&index) {
                self.parameters[index - 1]
            } else if !used[index] || self.handles[index].is_some() || panics.is_literal(index) {
                ir::Type::Unit
            } else if discriminants[index] {
                ir::Type::Scalar(ir::Scalar::Bool)
            } else if crate::types::has_reference(self.normalize_type(local.ty)) {
                pending[index] = true;
                ir::Type::Unit
            } else {
                self.lower_type(local.ty, local.source_info.span)?
            };
            self.locals.push(ty);
        }
        self.infer_references(body, pending)?;
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
            parameters: body.arg_count,
            locals: self.locals,
            blocks,
            source,
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
                mir::StatementKind::Assign(assignment)
                    if assignment
                        .0
                        .as_local()
                        .is_some_and(|local| self.handles[local.as_usize()].is_some()) => {}
                mir::StatementKind::Assign(assignment) => {
                    let (destination, value) = assignment.as_ref();
                    statements.push(ir::Statement {
                        destination: self.place(destination, span)?,
                        value: self.expression(
                            value,
                            destination.ty(&body.local_decls, self.module.tcx).ty,
                            span,
                        )?,
                        source: location(self.module.tcx, span),
                    });
                }
                mir::StatementKind::StorageLive(_)
                | mir::StatementKind::StorageDead(_)
                | mir::StatementKind::Nop => {}
                other => return Err((span, format!("unsupported MIR statement: {other:?}"))),
            }
        }
        let term = block.terminator();
        let span = term.source_info.span;
        // Checks report where Rust's `#[track_caller]` panics would: the outermost macro call.
        let caller = span.ctxt().outer_expn().expansion_cause().unwrap_or(span);
        let mut source = location(self.module.tcx, span);
        if let Some(message) = panics.message(self.module.tcx, &term.kind) {
            return Ok(ir::Block {
                statements,
                terminator: ir::Terminator::Panic(message),
                source: location(self.module.tcx, caller),
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
                source = location(self.module.tcx, caller);
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
            } if self.is_range_next(func, body) => self.range_next(
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
            } if self.is_subslice(func, body, span)? => self.subslice(
                &mut statements,
                args,
                destination,
                target.as_usize(),
                body,
                caller,
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
            } => {
                statements.push(ir::Statement {
                    destination: self.place(destination, span)?,
                    value: self.call(func, args, body, span)?,
                    source: source.clone(),
                });
                ir::Terminator::Goto(target.as_usize())
            }
            mir::TerminatorKind::Drop { place, target, .. } => {
                let ty = self.normalize_type(place.ty(&body.local_decls, self.module.tcx).ty);
                if ty.needs_drop(
                    self.module.tcx,
                    rustc_middle::ty::TypingEnv::fully_monomorphized(),
                ) {
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

impl FunctionImporter<'_, '_> {
    pub(super) fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    /// Appends a block after the MIR blocks and returns its id.
    pub(super) fn append(&mut self, body: &mir::Body<'_>, block: ir::Block) -> usize {
        self.appended.push(block);
        body.basic_blocks.len() + self.appended.len() - 1
    }
}
