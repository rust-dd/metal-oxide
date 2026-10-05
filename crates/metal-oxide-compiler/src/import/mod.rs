mod values;

use std::collections::HashMap;

use metal_oxide_ir as ir;
use rustc_middle::{
    mir,
    ty::{Instance, TyCtxt, consts::ConstExt},
};
use rustc_span::Span;

type Result<T> = std::result::Result<T, (Span, String)>;

pub(crate) fn module<'tcx>(
    tcx: TyCtxt<'tcx>,
    entries: &[Instance<'tcx>],
    instances: &[Instance<'tcx>],
) -> Result<ir::Module> {
    if entries.is_empty() {
        return Err((
            rustc_span::DUMMY_SP,
            "no kernel entrypoints were found".into(),
        ));
    }
    let instances = instances
        .iter()
        .copied()
        .filter(|i| crate::intrinsics::builtin(tcx, i.def_id()).is_none())
        .collect::<Vec<_>>();
    let ids = instances
        .iter()
        .enumerate()
        .map(|(id, &i)| (i, id))
        .collect::<HashMap<_, _>>();
    let functions = instances
        .iter()
        .map(|&instance| {
            let body = tcx.instance_mir(instance.def);
            let source = location(tcx, tcx.def_span(instance.def_id()));
            let mut context = Context {
                tcx,
                instance,
                ids: &ids,
                locals: Vec::new(),
                allocations: 0,
            };
            for local in body.local_decls.iter() {
                let ty = context.normalize_type(local.ty);
                context.locals.push(context.ty(ty, local.source_info.span)?);
            }
            let mut blocks = Vec::new();
            for block in body.basic_blocks.iter() {
                blocks.push(context.block(block, body)?);
            }
            let kernel = entries.contains(&instance);
            Ok(ir::Function {
                name: if kernel {
                    tcx.item_name(instance.def_id()).to_string()
                } else {
                    instance.to_string()
                },
                kernel,
                required_block: if kernel {
                    crate::collect::block_shape(tcx, instance.def_id())?
                } else {
                    None
                },
                parameters: body.arg_count,
                locals: context.locals,
                blocks,
                source,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ir::Module { functions })
}

struct Context<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    ids: &'a HashMap<Instance<'tcx>, usize>,
    locals: Vec<ir::Type>,
    allocations: u32,
}

impl<'tcx> Context<'_, 'tcx> {
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
                    statements.push(ir::Statement {
                        destination: self.destination(destination, span)?,
                        value: self.expression(value, span)?,
                        source: location(self.tcx, span),
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
        let source = location(self.tcx, span);
        let terminator = match &term.kind {
            mir::TerminatorKind::Goto { target } => ir::Terminator::Goto(target.as_usize()),
            mir::TerminatorKind::Return => ir::Terminator::Return,
            mir::TerminatorKind::Unreachable => ir::Terminator::Unreachable,
            mir::TerminatorKind::SwitchInt { discr, targets } => {
                let cases = targets.iter().collect::<Vec<_>>();
                if cases.len() != 1 {
                    return Err((span, "multi-way MIR switches are not supported yet".into()));
                }
                let (value, target) = cases[0];
                let operand = self.operand(discr, span)?;
                let ty = self.ty(
                    self.normalize_type(discr.ty(&body.local_decls, self.tcx)),
                    span,
                )?;
                if ty == ir::Type::Scalar(ir::Scalar::Bool) {
                    match value {
                        0 => ir::Terminator::Branch {
                            condition: operand,
                            then_block: targets.otherwise().as_usize(),
                            else_block: target.as_usize(),
                        },
                        1 => ir::Terminator::Branch {
                            condition: operand,
                            then_block: target.as_usize(),
                            else_block: targets.otherwise().as_usize(),
                        },
                        _ => return Err((span, "invalid bool switch value".into())),
                    }
                } else {
                    let constant = match ty {
                        ir::Type::Scalar(ir::Scalar::U32) => ir::Constant::U32(value as u32),
                        ir::Type::Scalar(ir::Scalar::I32) => ir::Constant::I32(value as i32),
                        _ => return Err((span, "unsupported switch discriminant type".into())),
                    };
                    let local = self.locals.len();
                    self.locals.push(ir::Type::Scalar(ir::Scalar::Bool));
                    statements.push(ir::Statement {
                        destination: local,
                        value: ir::Expression::Binary(
                            ir::BinaryOp::Eq,
                            operand,
                            ir::Operand::Constant(constant),
                        ),
                        source: source.clone(),
                    });
                    ir::Terminator::Branch {
                        condition: ir::Operand::local(local),
                        then_block: target.as_usize(),
                        else_block: targets.otherwise().as_usize(),
                    }
                }
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
                enabled: self.tcx.sess.overflow_checks() || !msg.is_optional_overflow_check(),
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
                let ty = self.normalize_type(place.ty(&body.local_decls, self.tcx).ty);
                if ty.needs_drop(self.tcx, rustc_middle::ty::TypingEnv::fully_monomorphized()) {
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

    fn call(
        &mut self,
        func: &mir::Operand<'tcx>,
        args: &[rustc_span::Spanned<mir::Operand<'tcx>>],
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Expression> {
        use rustc_middle::ty::{self, TypingEnv};
        let ty = self.normalize_type(func.ty(&body.local_decls, self.tcx));
        let ty::FnDef(def, args_types) = *ty.kind() else {
            return Err((span, "indirect calls cannot be imported".into()));
        };
        let args_types = self.tcx.instantiate_bound_regions_with_erased(args_types);
        let instance =
            Instance::try_resolve(self.tcx, TypingEnv::fully_monomorphized(), def, args_types)
                .map_err(|_| (span, "cannot resolve concrete function".into()))?
                .ok_or((span, "cannot resolve concrete function".into()))?;
        let arguments = args
            .iter()
            .map(|a| self.operand(&a.node, a.span))
            .collect::<Result<Vec<_>>>()?;
        if let Some(builtin) = crate::intrinsics::builtin(self.tcx, def) {
            return match (builtin, arguments.as_slice()) {
                ("threadgroup_alloc", []) => {
                    let ir::Type::Scalar(element) = self.ty(args_types.type_at(0), span)? else {
                        return Err((span, "threadgroup elements must be scalars".into()));
                    };
                    let length = args_types
                        .const_at(1)
                        .try_to_target_usize(self.tcx)
                        .and_then(|n| u32::try_from(n).ok())
                        .ok_or((
                            span,
                            "threadgroup length must be a concrete u32-sized constant".into(),
                        ))?;
                    let id = self.allocations;
                    self.allocations += 1;
                    Ok(ir::Expression::ThreadgroupAlloc {
                        id,
                        element,
                        length,
                    })
                }
                ("threadgroup_barrier", []) => Ok(ir::Expression::ThreadgroupBarrier),
                ("atomic_add", [buffer, index, value]) => Ok(ir::Expression::AtomicAdd {
                    buffer: buffer.clone(),
                    index: index.clone(),
                    value: value.clone(),
                }),
                ("buffer_load", [buffer, index]) => Ok(ir::Expression::BufferLoad {
                    buffer: buffer.clone(),
                    index: index.clone(),
                }),
                ("buffer_store", [buffer, index, value]) => Ok(ir::Expression::BufferStore {
                    buffer: buffer.clone(),
                    index: index.clone(),
                    value: value.clone(),
                }),
                ("thread_idx", []) => Ok(ir::Expression::Coordinates(ir::Builtin::ThreadIdx)),
                ("block_idx", []) => Ok(ir::Expression::Coordinates(ir::Builtin::BlockIdx)),
                ("block_dim", []) => Ok(ir::Expression::Coordinates(ir::Builtin::BlockDim)),
                ("grid_dim", []) => Ok(ir::Expression::Coordinates(ir::Builtin::GridDim)),
                _ => Err((span, "invalid device builtin arguments".into())),
            };
        }
        let function = *self
            .ids
            .get(&instance)
            .ok_or((span, "missing concrete function instance".into()))?;
        Ok(ir::Expression::Call {
            function,
            arguments,
        })
    }
}

fn location(tcx: TyCtxt<'_>, span: Span) -> ir::SourceLocation {
    let position = tcx.sess.source_map().lookup_char_pos(span.lo());
    ir::SourceLocation {
        file: position
            .file
            .name
            .prefer_remapped_unconditionally()
            .to_string(),
        line: position.line as u32,
        column: position.col.0 as u32 + 1,
    }
}
