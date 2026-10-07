use metal_oxide_ir as ir;
use rustc_middle::{
    mir,
    ty::{self, EarlyBinder, Instance, TyCtxt, TypingEnv},
};

pub(super) fn literal<'tcx>(tcx: TyCtxt<'tcx>, constant: mir::Const<'tcx>) -> Option<ir::Constant> {
    if constant.ty().is_unit() {
        return Some(ir::Constant::Unit);
    }
    let bits = constant.try_eval_bits(tcx, TypingEnv::fully_monomorphized())?;
    Some(match constant.ty().kind() {
        ty::Bool => ir::Constant::Bool(bits != 0),
        ty::Uint(ty::UintTy::U32) => ir::Constant::U32(bits as u32),
        ty::Uint(ty::UintTy::Usize) => ir::Constant::U32(u32::try_from(bits).ok()?),
        ty::Int(ty::IntTy::I32) => ir::Constant::I32(bits as i32),
        ty::Uint(ty::UintTy::U8) => ir::Constant::U8(bits as u8),
        ty::Uint(ty::UintTy::U16) => ir::Constant::U16(bits as u16),
        ty::Float(ty::FloatTy::F32) => ir::Constant::F32(bits as u32),
        _ => return None,
    })
}

fn operand<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    value: &mir::Operand<'tcx>,
    facts: &[Option<ir::Constant>],
) -> Option<ir::Constant> {
    match value {
        mir::Operand::Constant(value) => literal(
            tcx,
            instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(tcx, value.const_),
            ),
        ),
        mir::Operand::Copy(place) | mir::Operand::Move(place) if place.projection.is_empty() => {
            facts[place.local.as_usize()]
        }
        _ => None,
    }
}

fn integer(value: ir::Constant) -> Option<i64> {
    Some(match value {
        ir::Constant::U32(n) => i64::from(n),
        ir::Constant::I32(n) => i64::from(n),
        ir::Constant::U8(n) => i64::from(n),
        ir::Constant::U16(n) => i64::from(n),
        ir::Constant::Bool(n) => i64::from(n),
        _ => return None,
    })
}

pub(super) fn locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &mir::Body<'tcx>,
) -> Vec<Option<ir::Constant>> {
    let mut writes = vec![Vec::new(); body.local_decls.len()];
    let mut projected = vec![false; writes.len()];
    for block in body.basic_blocks.iter() {
        for statement in &block.statements {
            if let mir::StatementKind::Assign(value) = &statement.kind {
                let (place, value) = value.as_ref();
                writes[place.local.as_usize()].push(value);
                projected[place.local.as_usize()] |= !place.projection.is_empty();
            }
        }
        if let mir::TerminatorKind::Call { destination, .. } = &block.terminator().kind {
            projected[destination.local.as_usize()] = true;
        }
    }
    let mut facts = vec![None; writes.len()];
    loop {
        let mut changed = false;
        for (id, values) in writes.iter().enumerate() {
            if id <= body.arg_count || projected[id] || values.is_empty() {
                continue;
            }
            let evaluate = |value: &&mir::Rvalue<'tcx>| match value {
                mir::Rvalue::Use(value, _) => operand(tcx, instance, value, &facts),
                mir::Rvalue::BinaryOp(op, values) => {
                    let a = operand(tcx, instance, &values.0, &facts)?;
                    let b = operand(tcx, instance, &values.1, &facts)?;
                    if a.ty() != b.ty() {
                        return None;
                    }
                    let a = integer(a)?;
                    let b = integer(b)?;
                    Some(ir::Constant::Bool(match op {
                        mir::BinOp::Eq => a == b,
                        mir::BinOp::Ne => a != b,
                        mir::BinOp::Lt => a < b,
                        mir::BinOp::Le => a <= b,
                        mir::BinOp::Gt => a > b,
                        mir::BinOp::Ge => a >= b,
                        _ => return None,
                    }))
                }
                _ => None,
            };
            let first = evaluate(&values[0]);
            let fact = if values.iter().all(|value| evaluate(value) == first) {
                first
            } else {
                None
            };
            if facts[id] != fact {
                facts[id] = fact;
                changed = true;
            }
        }
        if !changed {
            return facts;
        }
    }
}
