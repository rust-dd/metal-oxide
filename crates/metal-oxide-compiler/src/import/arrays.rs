use metal_oxide_ir as ir;
use rustc_middle::{
    mir,
    ty::{self, Instance},
};
use rustc_span::{Span, Spanned};

use super::{
    FunctionKey, Result, calls::returns_no_references, function::FunctionImporter, location,
};

impl<'tcx> FunctionImporter<'_, 'tcx> {
    /// Builds the array that `<[T; N]>::map` (`mapped`) or `array::from_fn` returns: a loop
    /// passes each element, or each index, to the closure and stores what it returns. Returns
    /// the current block's terminator; the loop continues at `target`.
    pub(super) fn array_from_closure(
        &mut self,
        statements: &mut Vec<ir::Statement>,
        args: &[Spanned<mir::Operand<'tcx>>],
        (callee, mapped): (Instance<'tcx>, bool),
        (destination, target): (&mir::Place<'tcx>, usize),
        body: &mir::Body<'tcx>,
        span: Span,
    ) -> Result<ir::Terminator> {
        let tcx = self.module.tcx;
        let (closure, input) = crate::intrinsics::array_closure(tcx, callee, mapped)
            .ok_or((span, "array map and from_fn take a closure".into()))?;
        let (result, result_type) = self.typed_place(destination, span)?;
        let (element, length) = match result_type {
            ir::Type::Aggregate(id) => match self.module.types.table().get(id) {
                Some(ir::Aggregate::Array { element, length }) => (*element, u64::from(*length)),
                _ => return Err((span, "array map and from_fn build an array".into())),
            },
            _ => return Err((span, "array map and from_fn build an array".into())),
        };
        let source = location(tcx, span);
        let statement = |destination: ir::Place, value| ir::Statement {
            destination,
            value,
            source: source.clone(),
        };
        // The closure's environment, by reference unless it captures nothing.
        let environment = args.last().expect("a closure argument");
        let captures = self.operand_type(&environment.node, environment.span)?;
        let (environment_type, environment) = if captures == ir::Type::Unit {
            (ir::Type::Unit, ir::Operand::Constant(ir::Constant::Unit))
        } else {
            let ir::Operand::Place(captured) = self.operand(&environment.node, environment.span)?
            else {
                return Err((span, "array map and from_fn take a closure".into()));
            };
            let shared = matches!(
                closure
                    .instantiate_mir_and_normalize_erasing_regions(
                        tcx,
                        ty::TypingEnv::fully_monomorphized(),
                        ty::EarlyBinder::bind(
                            tcx,
                            crate::monomorphize::body(tcx, closure).local_decls
                                [mir::Local::from_usize(1)]
                            .ty,
                        ),
                    )
                    .kind(),
                ty::Ref(_, _, ty::Mutability::Not)
            );
            let pointer = self.module.types.intern_pointer(ir::Pointer {
                target: captures,
                space: ir::AddressSpace::Thread,
                mutable: !shared,
                slice: false,
            });
            let reference = ir::Place::local(self.temporary(pointer));
            statements.push(statement(
                reference.clone(),
                ir::Expression::Ref {
                    place: ir::Operand::Place(captured),
                    mutable: !shared,
                },
            ));
            (pointer, ir::Operand::Place(reference))
        };
        let input_type = self.lower_type(input, span)?;
        let returns = (!self.module.types.table().is_owned(element)
            && returns_no_references(tcx, closure))
        .then_some(element);
        let function = self.module.function(
            FunctionKey {
                instance: closure,
                entry: None,
                parameters: vec![environment_type, input_type],
                messages: vec![None, None],
                caller: None,
                returns,
            },
            false,
        )?;
        let index = ir::Place::local(self.temporary(ir::Type::Scalar(ir::Scalar::Usize)));
        let array = ir::Place::local(self.temporary(result_type));
        let at = |place: &ir::Place| {
            let mut place = place.clone();
            place.projection.push(ir::Projection::Index(index.local));
            place
        };
        let argument = if mapped {
            let mapped = match self.operand(&args[0].node, args[0].span)? {
                ir::Operand::Place(place) => place,
                constant => {
                    let ty = self.operand_type(&args[0].node, args[0].span)?;
                    let copy = ir::Place::local(self.temporary(ty));
                    statements.push(statement(copy.clone(), ir::Expression::Use(constant)));
                    copy
                }
            };
            ir::Operand::Place(at(&mapped))
        } else {
            ir::Operand::Place(index.clone())
        };
        statements.push(statement(
            array.clone(),
            ir::Expression::Uninit(result_type),
        ));
        statements.push(statement(
            index.clone(),
            ir::Expression::Use(ir::Operand::Constant(ir::Constant::Usize(0))),
        ));
        let header = body.basic_blocks.len() + self.appended.len();
        let condition = ir::Place::local(self.temporary(ir::Type::Scalar(ir::Scalar::Bool)));
        let iteration = vec![
            statement(
                at(&array),
                ir::Expression::Call {
                    function,
                    arguments: vec![environment, argument],
                },
            ),
            statement(
                index.clone(),
                ir::Expression::Binary(
                    ir::BinaryOp::Add,
                    ir::Operand::Place(index.clone()),
                    ir::Operand::Constant(ir::Constant::Usize(1)),
                ),
            ),
        ];
        let blocks = [
            ir::Block {
                statements: vec![statement(
                    condition.clone(),
                    ir::Expression::Binary(
                        ir::BinaryOp::Lt,
                        ir::Operand::Place(index.clone()),
                        ir::Operand::Constant(ir::Constant::Usize(length)),
                    ),
                )],
                terminator: ir::Terminator::Branch {
                    condition: ir::Operand::Place(condition),
                    then_block: header + 1,
                    else_block: header + 2,
                },
                source: source.clone(),
            },
            ir::Block {
                statements: iteration,
                terminator: ir::Terminator::Goto(header),
                source: source.clone(),
            },
            ir::Block {
                statements: vec![statement(
                    result,
                    ir::Expression::Use(ir::Operand::Place(array)),
                )],
                terminator: ir::Terminator::Goto(target),
                source,
            },
        ];
        for block in blocks {
            self.append(body, block);
        }
        Ok(ir::Terminator::Goto(header))
    }
}
