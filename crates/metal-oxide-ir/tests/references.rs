use metal_oxide_ir::*;

fn source() -> SourceLocation {
    SourceLocation {
        file: "kernel.rs".into(),
        line: 7,
        column: 5,
    }
}

fn statement(destination: Place, value: Expression) -> Statement {
    Statement {
        destination,
        value,
        source: source(),
    }
}

fn assign(local: usize, value: Expression) -> Statement {
    statement(Place::local(local), value)
}

fn block(statements: Vec<Statement>, terminator: Terminator) -> Block {
    Block {
        statements,
        terminator,
        source: source(),
    }
}

fn helper(types: TypeTable, locals: Vec<Type>, blocks: Vec<Block>) -> Module {
    Module {
        types,
        functions: vec![Function {
            name: "helper".into(),
            kernel: false,
            required_block: None,
            parameters: 0,
            locals,
            blocks,
            source: source(),
        }],
    }
}

fn deref(local: usize) -> Place {
    Place {
        local,
        projection: vec![Projection::Deref],
    }
}

fn reference(place: Place, mutable: bool) -> Expression {
    Expression::Ref {
        place: Operand::Place(place),
        mutable,
    }
}

fn usize(value: u64) -> Operand {
    Operand::Constant(Constant::Usize(value))
}

const U32: Type = Type::Scalar(Scalar::U32);
const USIZE: Type = Type::Scalar(Scalar::Usize);
const BOOL: Type = Type::Scalar(Scalar::Bool);

fn pointer(types: &mut TypeTable, target: Type, mutable: bool, slice: bool) -> Type {
    types.intern_pointer(Pointer {
        target,
        space: AddressSpace::Thread,
        mutable,
        slice,
    })
}

#[test]
fn a_write_through_a_shared_reference_is_rejected() {
    let mut types = TypeTable::default();
    let shared = pointer(&mut types, U32, false, false);
    let module = helper(
        types,
        vec![Type::Unit, U32, shared],
        vec![block(
            vec![
                assign(1, Expression::Use(Operand::Constant(Constant::U32(1)))),
                assign(2, reference(Place::local(1), false)),
                statement(
                    deref(2),
                    Expression::Use(Operand::Constant(Constant::U32(2))),
                ),
            ],
            Terminator::Return,
        )],
    );
    let error = validate(&module).unwrap_err();
    assert!(error.message.contains("shared reference"), "{error}");
}

#[test]
fn a_mutable_reborrow_of_a_shared_reference_is_rejected() {
    let mut types = TypeTable::default();
    let shared = pointer(&mut types, U32, false, false);
    let unique = pointer(&mut types, U32, true, false);
    let module = helper(
        types,
        vec![Type::Unit, U32, shared, unique],
        vec![block(
            vec![
                assign(1, Expression::Use(Operand::Constant(Constant::U32(1)))),
                assign(2, reference(Place::local(1), false)),
                assign(3, reference(deref(2), true)),
            ],
            Terminator::Return,
        )],
    );
    let error = validate(&module).unwrap_err();
    assert!(error.message.contains("mutable path"), "{error}");
}

/// Locals 1-3 hold `[7, 8]`, a reference to it and its slice; local 4 is an index of 1.
fn slice_of_pair(types: &mut TypeTable) -> (Vec<Type>, Vec<Statement>) {
    let array = types.intern(Aggregate::Array {
        element: U32,
        length: 2,
    });
    let reference_type = pointer(types, array, false, false);
    let slice = pointer(types, U32, false, true);
    let Type::Aggregate(id) = array else {
        unreachable!()
    };
    (
        vec![U32, array, reference_type, slice, USIZE],
        vec![
            assign(
                1,
                Expression::Aggregate {
                    ty: id,
                    fields: vec![
                        Operand::Constant(Constant::U32(7)),
                        Operand::Constant(Constant::U32(8)),
                    ],
                },
            ),
            assign(2, reference(Place::local(1), false)),
            assign(3, Expression::Slice(Operand::local(2))),
            assign(4, Expression::Use(usize(1))),
        ],
    )
}

fn element(slice: usize, index: usize) -> Operand {
    Operand::Place(Place {
        local: slice,
        projection: vec![Projection::Deref, Projection::Index(index)],
    })
}

#[test]
fn a_parameter_slice_index_needs_a_length_check() {
    let mut types = TypeTable::default();
    let slice = pointer(&mut types, U32, false, true);
    let mut module = helper(
        types,
        vec![U32, USIZE, USIZE, slice, USIZE],
        vec![block(
            vec![
                assign(4, Expression::Use(usize(1))),
                assign(0, Expression::Use(element(3, 4))),
            ],
            Terminator::Return,
        )],
    );
    module.functions[0].parameters = 3;
    let error = validate(&module).unwrap_err();
    assert!(error.message.contains("checked slice length"), "{error}");
}

/// Locals 5 and 6 check index 4 against the length of slice 3; block 1 reads the element.
fn checked_read(locals: &mut Vec<Type>, statements: &mut Vec<Statement>) -> Vec<Block> {
    locals.extend([USIZE, BOOL]);
    statements.extend([
        assign(5, Expression::Len(Operand::local(3))),
        assign(
            6,
            Expression::Binary(BinaryOp::Lt, Operand::local(4), Operand::local(5)),
        ),
    ]);
    vec![
        block(
            std::mem::take(statements),
            Terminator::Assert {
                condition: Operand::local(6),
                expected: true,
                enabled: true,
                target: 1,
                check: Check::Bounds,
            },
        ),
        block(
            vec![assign(0, Expression::Use(element(3, 4)))],
            Terminator::Return,
        ),
    ]
}

#[test]
fn an_array_slice_index_is_proved_by_the_array_length() {
    let mut types = TypeTable::default();
    let (mut locals, mut statements) = slice_of_pair(&mut types);
    let blocks = checked_read(&mut locals, &mut statements);
    let analyses = validate(&helper(types, locals, blocks)).unwrap();
    assert!(analyses[0].checks.is_empty());
}

#[test]
fn a_parameter_slice_index_is_checked_on_the_gpu() {
    let mut types = TypeTable::default();
    let slice = pointer(&mut types, U32, false, true);
    let mut locals = vec![U32, USIZE, USIZE, slice, USIZE];
    let mut statements = vec![assign(4, Expression::Use(usize(1)))];
    let blocks = checked_read(&mut locals, &mut statements);
    let mut module = helper(types, locals, blocks);
    module.functions[0].parameters = 3;
    let analyses = validate(&module).unwrap();
    assert_eq!(analyses[0].checks.iter().copied().collect::<Vec<_>>(), [0]);
}

/// `x = 1; *(&mut x) = 7; [10, 20][x]`: the write through the pointer must void the fact x = 1.
#[test]
fn a_write_through_a_reference_voids_facts_about_the_borrowed_local() {
    let mut types = TypeTable::default();
    let unique = pointer(&mut types, USIZE, true, false);
    let array = types.intern(Aggregate::Array {
        element: U32,
        length: 2,
    });
    let Type::Aggregate(id) = array else {
        unreachable!()
    };
    let module = helper(
        types,
        vec![U32, USIZE, unique, array, BOOL],
        vec![
            block(
                vec![
                    assign(1, Expression::Use(usize(1))),
                    assign(2, reference(Place::local(1), true)),
                    statement(deref(2), Expression::Use(usize(7))),
                    assign(
                        3,
                        Expression::Aggregate {
                            ty: id,
                            fields: vec![
                                Operand::Constant(Constant::U32(10)),
                                Operand::Constant(Constant::U32(20)),
                            ],
                        },
                    ),
                    assign(
                        4,
                        Expression::Binary(BinaryOp::Lt, Operand::local(1), usize(2)),
                    ),
                ],
                Terminator::Assert {
                    condition: Operand::local(4),
                    expected: true,
                    enabled: true,
                    target: 1,
                    check: Check::Bounds,
                },
            ),
            block(
                vec![assign(
                    0,
                    Expression::Use(Operand::Place(Place {
                        local: 3,
                        projection: vec![Projection::Index(1)],
                    })),
                )],
                Terminator::Return,
            ),
        ],
    );
    let analyses = validate(&module).unwrap();
    assert_eq!(analyses[0].checks.iter().copied().collect::<Vec<_>>(), [0]);
}

#[test]
fn a_buffer_slice_outside_a_kernel_is_rejected() {
    let mut types = TypeTable::default();
    let slice = pointer(&mut types, U32, false, true);
    let buffer = Type::Buffer {
        element: Element::Scalar(Scalar::U32),
        access: Access::Read,
        address_space: AddressSpace::Device,
    };
    let device_slice = types.intern_pointer(Pointer {
        target: U32,
        space: AddressSpace::Device,
        mutable: false,
        slice: true,
    });
    let mut module = helper(
        types,
        vec![Type::Unit, buffer, device_slice, slice],
        vec![block(
            vec![assign(2, Expression::Slice(Operand::local(1)))],
            Terminator::Return,
        )],
    );
    module.functions[0].parameters = 1;
    let error = validate(&module).unwrap_err();
    assert!(error.message.contains("kernel buffer parameter"), "{error}");
}
