use super::Used;
use metal_oxide_ir::*;

const U32: Type = Type::Scalar(Scalar::U32);
const USIZE: Type = Type::Scalar(Scalar::Usize);
const BOOL: Type = Type::Scalar(Scalar::Bool);

fn source() -> SourceLocation {
    SourceLocation {
        file: "unused.rs".into(),
        line: 1,
        column: 1,
    }
}

fn function(parameters: usize, locals: Vec<Type>, blocks: Vec<Block>) -> Function {
    Function {
        name: "helper".into(),
        kernel: false,
        required_block: None,
        requires: Vec::new(),
        parameters,
        locals,
        blocks,
        source: source(),
    }
}

fn block(statements: Vec<Statement>, terminator: Terminator) -> Block {
    Block {
        statements,
        terminator,
        source: source(),
    }
}

fn assign(local: usize, value: Expression) -> Statement {
    store(Place::local(local), value)
}

fn store(destination: Place, value: Expression) -> Statement {
    Statement {
        destination,
        value,
        source: source(),
    }
}

fn value(local: usize) -> Expression {
    Expression::Use(Operand::local(local))
}

fn uint(value: u32) -> Expression {
    Expression::Use(Operand::Constant(Constant::U32(value)))
}

fn index(value: u64) -> Expression {
    Expression::Use(Operand::Constant(Constant::Usize(value)))
}

fn reference(place: Place) -> Expression {
    Expression::Ref {
        place: Operand::Place(place),
        mutable: true,
    }
}

fn projected(local: usize, projection: Vec<Projection>) -> Place {
    Place { local, projection }
}

fn pointer(types: &mut TypeTable, target: Type, slice: bool) -> Type {
    types.intern_pointer(Pointer {
        target,
        space: AddressSpace::Thread,
        mutable: true,
        slice,
    })
}

fn array(types: &mut TypeTable) -> Type {
    types.intern(Aggregate::Array {
        element: U32,
        length: 2,
    })
}

fn aggregate(ty: Type, fields: Vec<Operand>) -> Expression {
    let Type::Aggregate(ty) = ty else {
        unreachable!()
    };
    Expression::Aggregate { ty, fields }
}

fn array_value(ty: Type, a: u32, b: u32) -> Expression {
    aggregate(
        ty,
        [a, b]
            .map(|value| Operand::Constant(Constant::U32(value)))
            .to_vec(),
    )
}

fn used(types: TypeTable, functions: Vec<Function>) -> Used {
    let module = Module { types, functions };
    let analyses = validate(&module).unwrap();
    Used::new(&module, &module.functions[0], &analyses[0])
}

#[test]
fn dead_scalar_and_array_cycles_have_no_roots() {
    let mut types = TypeTable::default();
    let array = array(&mut types);
    let used = used(
        types,
        vec![function(
            0,
            vec![U32, U32, U32, array, USIZE],
            vec![block(
                vec![
                    assign(1, uint(7)),
                    assign(2, value(1)),
                    assign(1, value(2)),
                    assign(3, array_value(array, 1, 2)),
                    assign(4, index(0)),
                    store(
                        projected(3, vec![Projection::Index(4)]),
                        Expression::Use(Operand::Place(projected(3, vec![Projection::Index(4)]))),
                    ),
                    assign(0, uint(42)),
                ],
                Terminator::Return,
            )],
        )],
    );
    assert_eq!(
        used.statements[0],
        [false, false, false, false, false, false, true]
    );
    assert_eq!(used.locals, [true, false, false, false, false]);
}

#[test]
fn a_full_overwrite_kills_earlier_scalar_and_aggregate_values() {
    let mut types = TypeTable::default();
    let array = array(&mut types);
    let used = used(
        types,
        vec![function(
            0,
            vec![array, array, U32, USIZE],
            vec![block(
                vec![
                    assign(2, uint(8)),
                    assign(2, uint(9)),
                    assign(1, array_value(array, 1, 2)),
                    assign(3, index(0)),
                    store(projected(1, vec![Projection::Index(3)]), value(2)),
                    assign(1, array_value(array, 3, 4)),
                    assign(0, value(1)),
                ],
                Terminator::Return,
            )],
        )],
    );
    assert_eq!(
        used.statements[0],
        [false, false, false, false, false, true, true]
    );
}

#[test]
fn taking_an_address_does_not_read_the_old_value() {
    let mut types = TypeTable::default();
    let pointer = pointer(&mut types, U32, false);
    let used = used(
        types,
        vec![function(
            0,
            vec![U32, U32, pointer],
            vec![block(
                vec![
                    assign(1, uint(1)),
                    assign(2, reference(Place::local(1))),
                    assign(1, uint(2)),
                    assign(
                        0,
                        Expression::Use(Operand::Place(projected(2, vec![Projection::Deref]))),
                    ),
                ],
                Terminator::Return,
            )],
        )],
    );
    assert_eq!(used.statements[0], [false, true, true, true]);
}

#[test]
fn a_loop_reinitializes_an_array_after_its_last_borrow_use() {
    let mut types = TypeTable::default();
    let array = array(&mut types);
    let pointer = pointer(&mut types, array, false);
    let used = used(
        types,
        vec![
            function(
                1,
                vec![U32, BOOL, array, pointer, U32, USIZE],
                vec![
                    block(vec![], Terminator::Goto(1)),
                    block(
                        vec![
                            assign(2, array_value(array, 1, 2)),
                            assign(3, reference(Place::local(2))),
                            assign(
                                4,
                                Expression::Call {
                                    function: 1,
                                    arguments: vec![Operand::local(3)],
                                },
                            ),
                            assign(5, index(0)),
                            store(
                                projected(2, vec![Projection::Index(5)]),
                                Expression::Use(Operand::Place(projected(
                                    2,
                                    vec![Projection::Index(5)],
                                ))),
                            ),
                        ],
                        Terminator::Goto(2),
                    ),
                    block(
                        vec![],
                        Terminator::Branch {
                            condition: Operand::local(1),
                            then_block: 1,
                            else_block: 3,
                        },
                    ),
                    block(vec![assign(0, uint(0))], Terminator::Return),
                ],
            ),
            function(
                1,
                vec![U32, pointer, USIZE],
                vec![block(
                    vec![
                        assign(2, index(0)),
                        assign(
                            0,
                            Expression::Use(Operand::Place(projected(
                                1,
                                vec![Projection::Deref, Projection::Index(2)],
                            ))),
                        ),
                    ],
                    Terminator::Return,
                )],
            ),
        ],
    );
    assert_eq!(used.statements[1], [true, true, true, false, false]);
}

#[test]
fn liveness_crosses_empty_blocks_and_loop_backedges() {
    let used = used(
        TypeTable::default(),
        vec![function(
            1,
            vec![U32, BOOL, U32, U32],
            vec![
                block(
                    vec![assign(2, uint(7)), assign(3, uint(9))],
                    Terminator::Goto(3),
                ),
                block(vec![assign(2, value(3))], Terminator::Goto(2)),
                block(
                    vec![],
                    Terminator::Branch {
                        condition: Operand::local(1),
                        then_block: 3,
                        else_block: 4,
                    },
                ),
                block(vec![assign(3, value(2))], Terminator::Goto(1)),
                block(vec![assign(0, value(2))], Terminator::Return),
            ],
        )],
    );
    assert_eq!(used.statements[0], [true, false]);
    assert!(used.statement(1, 0));
    assert!(used.statement(3, 0));
}

#[test]
fn an_indirect_store_is_retained_without_reading_its_old_referent() {
    let mut types = TypeTable::default();
    let pointer = pointer(&mut types, U32, false);
    let used = used(
        types,
        vec![function(
            0,
            vec![U32, U32, pointer],
            vec![block(
                vec![
                    assign(1, uint(1)),
                    assign(2, reference(Place::local(1))),
                    store(projected(2, vec![Projection::Deref]), uint(2)),
                    assign(0, uint(0)),
                ],
                Terminator::Return,
            )],
        )],
    );
    assert_eq!(used.statements[0], [false, true, true, true]);
    assert!(used.local(1));
}

mod aliases;
