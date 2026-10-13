use metal_oxide_ir as ir;
use rustc_middle::{mir, ty::Instance};
use rustc_span::Span;

use super::{ModuleImporter, Result, control, location, panics::Panics, ranges};

pub(super) struct FunctionImporter<'a, 'tcx> {
    pub(super) module: &'a mut ModuleImporter<'tcx>,
    pub(super) instance: Instance<'tcx>,
    pub(super) locals: Vec<ir::Type>,
    pub(super) allocations: u32,
    pub(super) constants: Vec<Option<ir::Constant>>,
    pub(super) references: Vec<Option<mir::Local>>,
    /// Blocks numbered after the MIR blocks, from expanded range iteration.
    pub(super) appended: Vec<ir::Block>,
}

impl<'a, 'tcx> FunctionImporter<'a, 'tcx> {
    pub(super) fn new(module: &'a mut ModuleImporter<'tcx>, instance: Instance<'tcx>) -> Self {
        Self {
            module,
            instance,
            locals: Vec::new(),
            allocations: 0,
            constants: Vec::new(),
            references: Vec::new(),
            appended: Vec::new(),
        }
    }

    pub(super) fn import(mut self, kernel: bool) -> Result<ir::Function> {
        let tcx = self.module.tcx;
        let body = tcx.instance_mir(self.instance.def);
        let source = location(tcx, tcx.def_span(self.instance.def_id()));
        let used = control::used_locals(body);
        let panics = Panics::new(tcx, body)?;
        self.constants = super::constants::locals(tcx, self.instance, body);
        self.references = ranges::references(body);
        let discriminants = ranges::discriminants(body);
        for (index, local) in body.local_decls.iter().enumerate() {
            let ty = if !used[index] || self.references[index].is_some() || panics.is_literal(index)
            {
                ir::Type::Unit
            } else if discriminants[index] {
                ir::Type::Scalar(ir::Scalar::Bool)
            } else {
                self.lower_type(local.ty, local.source_info.span)?
            };
            self.locals.push(ty);
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
                        .is_some_and(|local| self.references[local.as_usize()].is_some()) => {}
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
