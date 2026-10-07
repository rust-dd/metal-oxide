use metal_oxide_ir as ir;
use rustc_middle::{mir, ty::Instance};

use super::{ModuleImporter, Result, control, location};

pub(super) struct FunctionImporter<'a, 'tcx> {
    pub(super) module: &'a mut ModuleImporter<'tcx>,
    pub(super) instance: Instance<'tcx>,
    pub(super) locals: Vec<ir::Type>,
    pub(super) allocations: u32,
}

impl<'a, 'tcx> FunctionImporter<'a, 'tcx> {
    pub(super) fn new(module: &'a mut ModuleImporter<'tcx>, instance: Instance<'tcx>) -> Self {
        Self {
            module,
            instance,
            locals: Vec::new(),
            allocations: 0,
        }
    }

    pub(super) fn import(mut self, kernel: bool) -> Result<ir::Function> {
        let tcx = self.module.tcx;
        let body = tcx.instance_mir(self.instance.def);
        let source = location(tcx, tcx.def_span(self.instance.def_id()));
        let used = control::used_locals(body);
        for (index, local) in body.local_decls.iter().enumerate() {
            let ty = if used[index] {
                self.lower_type(local.ty, local.source_info.span)?
            } else {
                ir::Type::Unit
            };
            self.locals.push(ty);
        }
        let mut blocks = Vec::with_capacity(body.basic_blocks.len());
        for block in body.basic_blocks.iter() {
            blocks.push(self.block(block, body)?);
        }
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
    ) -> Result<ir::Block> {
        let mut statements = Vec::new();
        for statement in &block.statements {
            let span = statement.source_info.span;
            match &statement.kind {
                mir::StatementKind::Assign(assignment) => {
                    let (destination, value) = assignment.as_ref();
                    let expression = self.expression(value, span)?;
                    let source = location(self.module.tcx, span);
                    if let [mir::ProjectionElem::Field(field, _)] =
                        destination.projection.as_slice()
                    {
                        let local = destination.local.as_usize();
                        if !matches!(self.locals[local], ir::Type::Record(_)) {
                            return Err((
                                span,
                                "field writes require an owned scalar record".into(),
                            ));
                        }
                        let ty = self.lower_type(
                            destination.ty(&body.local_decls, self.module.tcx).ty,
                            span,
                        )?;
                        let temporary = self.locals.len();
                        self.locals.push(ty);
                        statements.push(ir::Statement {
                            destination: temporary,
                            value: expression,
                            source: source.clone(),
                        });
                        statements.push(ir::Statement {
                            destination: local,
                            value: ir::Expression::RecordUpdate {
                                record: ir::Operand::local(local),
                                field: field.as_u32(),
                                value: ir::Operand::local(temporary),
                            },
                            source,
                        });
                    } else {
                        statements.push(ir::Statement {
                            destination: self.destination(destination, span)?,
                            value: expression,
                            source,
                        });
                    }
                }
                mir::StatementKind::StorageLive(_)
                | mir::StatementKind::StorageDead(_)
                | mir::StatementKind::Nop => {}
                other => return Err((span, format!("unsupported MIR statement: {other:?}"))),
            }
        }
        let term = block.terminator();
        let span = term.source_info.span;
        let source = location(self.module.tcx, span);
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
            } => ir::Terminator::Assert {
                condition: self.operand(cond, span)?,
                expected: *expected,
                enabled: self.module.tcx.sess.overflow_checks()
                    || !msg.is_optional_overflow_check(),
                target: target.as_usize(),
                message: format!("{msg:?}"),
            },
            mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target: Some(target),
                ..
            } => {
                statements.push(ir::Statement {
                    destination: self.destination(destination, span)?,
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
