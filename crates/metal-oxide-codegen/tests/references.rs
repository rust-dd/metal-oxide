#[allow(dead_code)]
mod support;

use metal_oxide_ir::*;
use support::{local, source};

fn statement(destination: Place, value: Expression) -> Statement {
    Statement {
        destination,
        value,
        source: source(),
    }
}

fn assign(destination: usize, value: Expression) -> Statement {
    statement(Place::local(destination), value)
}

fn block(statements: Vec<Statement>, terminator: Terminator) -> Block {
    Block {
        statements,
        terminator,
        source: source(),
    }
}

fn bounds(condition: usize, target: usize) -> Terminator {
    Terminator::Assert {
        condition: local(condition),
        expected: true,
        enabled: true,
        target,
        check: Check::Bounds,
    }
}

fn function(parameters: usize, locals: Vec<Type>, blocks: Vec<Block>) -> Function {
    Function {
        name: "helper".into(),
        kernel: false,
        required_block: None,
        parameters,
        locals,
        blocks,
        source: source(),
    }
}

fn elements(local: usize, index: usize) -> Place {
    Place {
        local,
        projection: vec![Projection::Deref, Projection::Index(index)],
    }
}

fn reference(local: usize, mutable: bool) -> Expression {
    Expression::Ref {
        place: Operand::local(local),
        mutable,
    }
}

const USIZE: Type = Type::Scalar(Scalar::Usize);
const F32: Type = Type::Scalar(Scalar::F32);

#[test]
fn a_helper_writes_through_a_reference_to_a_caller_local() {
    let mut types = TypeTable::default();
    let u32 = Type::Scalar(Scalar::U32);
    let pointer = types.intern_pointer(Pointer {
        target: u32,
        space: AddressSpace::Thread,
        mutable: true,
        slice: false,
    });
    let deref = Place {
        local: 1,
        projection: vec![Projection::Deref],
    };
    let module = Module {
        types,
        functions: vec![
            function(
                0,
                vec![u32, u32, pointer, Type::Unit],
                vec![block(
                    vec![
                        assign(1, Expression::Use(Operand::Constant(Constant::U32(41)))),
                        assign(2, reference(1, true)),
                        assign(
                            3,
                            Expression::Call {
                                function: 1,
                                arguments: vec![local(2)],
                            },
                        ),
                        assign(0, Expression::Use(local(1))),
                    ],
                    Terminator::Return,
                )],
            ),
            function(
                1,
                vec![Type::Unit, pointer, u32],
                vec![block(
                    vec![
                        assign(
                            2,
                            Expression::Binary(
                                BinaryOp::Add,
                                Operand::Place(deref.clone()),
                                Operand::Constant(Constant::U32(1)),
                            ),
                        ),
                        statement(deref, Expression::Use(local(2))),
                    ],
                    Terminator::Return,
                )],
            ),
        ],
    };
    assert_eq!(
        support::execute(&module, "std::cout << metal_oxide_fn_0(ctx);"),
        "42"
    );
}

/// `[1.0, 2.0, 3.0, 4.0]` behind a reference and a slice of it in locals 1-3.
fn array_slice(types: &mut TypeTable, mutable: bool) -> (Vec<Type>, Vec<Statement>) {
    let array = types.intern(Aggregate::Array {
        element: F32,
        length: 4,
    });
    let reference_type = types.intern_pointer(Pointer {
        target: array,
        space: AddressSpace::Thread,
        mutable,
        slice: false,
    });
    let slice = types.intern_pointer(Pointer {
        target: F32,
        space: AddressSpace::Thread,
        mutable,
        slice: true,
    });
    let values = [1.0_f32, 2.0, 3.0, 4.0]
        .map(|value| Operand::Constant(Constant::F32(value.to_bits())))
        .to_vec();
    let Type::Aggregate(id) = array else {
        unreachable!()
    };
    (
        vec![F32, array, reference_type, slice],
        vec![
            assign(
                1,
                Expression::Aggregate {
                    ty: id,
                    fields: values,
                },
            ),
            assign(2, reference(1, mutable)),
            assign(3, Expression::Slice(local(2))),
        ],
    )
}

#[test]
fn a_checked_slice_index_reads_the_array_element() {
    let mut types = TypeTable::default();
    let (mut locals, mut statements) = array_slice(&mut types, false);
    locals.extend([USIZE, USIZE, Type::Scalar(Scalar::Bool)]);
    statements.extend([
        assign(4, Expression::Len(local(3))),
        assign(5, Expression::Use(Operand::Constant(Constant::Usize(2)))),
        assign(6, Expression::Binary(BinaryOp::Lt, local(5), local(4))),
    ]);
    let module = Module {
        types,
        functions: vec![function(
            0,
            locals,
            vec![
                block(statements, bounds(6, 1)),
                block(
                    vec![assign(0, Expression::Use(Operand::Place(elements(3, 5))))],
                    Terminator::Return,
                ),
            ],
        )],
    };
    assert_eq!(
        support::execute(&module, "std::cout << metal_oxide_fn_0(ctx);"),
        "3"
    );
}

#[test]
fn a_subslice_starts_at_its_offset() {
    let mut types = TypeTable::default();
    let (mut locals, mut statements) = array_slice(&mut types, false);
    let usize = |value| Operand::Constant(Constant::Usize(value));
    locals.extend([locals[3], USIZE, USIZE, Type::Scalar(Scalar::Bool)]);
    statements.extend([
        assign(
            4,
            Expression::Subslice {
                slice: local(3),
                start: usize(1),
                end: usize(3),
            },
        ),
        assign(5, Expression::Len(local(4))),
        assign(6, Expression::Use(usize(1))),
        assign(7, Expression::Binary(BinaryOp::Lt, local(6), local(5))),
    ]);
    let module = Module {
        types,
        functions: vec![function(
            0,
            locals,
            vec![
                block(statements, bounds(7, 1)),
                block(
                    vec![assign(0, Expression::Use(Operand::Place(elements(4, 6))))],
                    Terminator::Return,
                ),
            ],
        )],
    };
    assert_eq!(
        support::execute(&module, "std::cout << metal_oxide_fn_0(ctx);"),
        "3"
    );
}

#[test]
fn a_mutable_slice_write_is_visible_through_a_shared_reborrow() {
    let mut types = TypeTable::default();
    let (mut locals, mut statements) = array_slice(&mut types, true);
    let shared = types.intern_pointer(Pointer {
        target: F32,
        space: AddressSpace::Thread,
        mutable: false,
        slice: true,
    });
    let bool = Type::Scalar(Scalar::Bool);
    locals.extend([USIZE, USIZE, bool, shared, USIZE, bool]);
    statements.extend([
        assign(4, Expression::Len(local(3))),
        assign(5, Expression::Use(Operand::Constant(Constant::Usize(1)))),
        assign(6, Expression::Binary(BinaryOp::Lt, local(5), local(4))),
    ]);
    let nine = Operand::Constant(Constant::F32(9.0_f32.to_bits()));
    let module = Module {
        types,
        functions: vec![function(
            0,
            locals,
            vec![
                block(statements, bounds(6, 1)),
                block(
                    vec![
                        statement(elements(3, 5), Expression::Use(nine)),
                        assign(
                            7,
                            Expression::Ref {
                                place: Operand::Place(Place {
                                    local: 3,
                                    projection: vec![Projection::Deref],
                                }),
                                mutable: false,
                            },
                        ),
                        assign(8, Expression::Len(local(7))),
                        assign(9, Expression::Binary(BinaryOp::Lt, local(5), local(8))),
                    ],
                    bounds(9, 2),
                ),
                block(
                    vec![assign(0, Expression::Use(Operand::Place(elements(7, 5))))],
                    Terminator::Return,
                ),
            ],
        )],
    };
    assert_eq!(
        support::execute(&module, "std::cout << metal_oxide_fn_0(ctx);"),
        "9"
    );
}
