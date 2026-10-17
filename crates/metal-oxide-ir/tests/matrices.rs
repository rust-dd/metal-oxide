use metal_oxide_ir::*;

fn source() -> SourceLocation {
    SourceLocation {
        file: "kernel.rs".into(),
        line: 3,
        column: 5,
    }
}

fn assign(local: usize, value: Expression) -> Statement {
    Statement {
        destination: Place::local(local),
        value,
        source: source(),
    }
}

fn uint(value: u32) -> Operand {
    Operand::Constant(Constant::U32(value))
}

fn matrix(op: MatrixOp, arguments: Vec<Operand>) -> Expression {
    Expression::Matrix { op, arguments }
}

fn buffer(element: Scalar, access: Access) -> Type {
    Type::Buffer {
        element: Element::Scalar(element),
        access,
        address_space: AddressSpace::Device,
    }
}

/// A kernel of one block over the locals `[unit, parameters.., locals..]`.
fn kernel(parameters: Vec<Type>, locals: Vec<Type>, statements: Vec<Statement>) -> Module {
    let count = parameters.len();
    Module {
        types: TypeTable::default(),
        functions: vec![Function {
            name: "kernel".into(),
            kernel: true,
            required_block: Some([64, 1, 1]),
            requires: Vec::new(),
            parameters: count,
            locals: [vec![Type::Unit], parameters, locals].concat(),
            blocks: vec![Block {
                statements,
                terminator: Terminator::Return,
                source: source(),
            }],
            source: source(),
        }],
    }
}

/// Loads an F16 block from `from`, accumulates its square in f32 and stores that to `to`.
fn product(from: Type, to: Type) -> Module {
    kernel(
        vec![from, to],
        vec![Type::Matrix(Scalar::F16), Type::Matrix(Scalar::F32)],
        vec![
            assign(
                3,
                matrix(
                    MatrixOp::Load { transposed: true },
                    vec![Operand::local(1), uint(0), uint(0), uint(8)],
                ),
            ),
            assign(
                4,
                matrix(MatrixOp::Splat, vec![Operand::Constant(Constant::F32(0))]),
            ),
            assign(
                4,
                matrix(
                    MatrixOp::MultiplyAccumulate,
                    vec![Operand::local(3), Operand::local(3), Operand::local(4)],
                ),
            ),
            assign(
                0,
                matrix(
                    MatrixOp::Store { transposed: false },
                    vec![
                        Operand::local(2),
                        uint(0),
                        uint(0),
                        uint(8),
                        Operand::local(4),
                    ],
                ),
            ),
        ],
    )
}

#[test]
fn matrix_operations_type_loads_products_and_stores() {
    let module = product(
        buffer(Scalar::F16, Access::Read),
        buffer(Scalar::F32, Access::Write),
    );
    let analyses = validate(&module).unwrap();
    assert!(analyses[0].cooperative && analyses[0].effects);
}

#[test]
fn matrix_operations_reject_mismatched_buffers_and_elements() {
    for (from, to) in [
        (
            buffer(Scalar::F16, Access::Write),
            buffer(Scalar::F32, Access::Write),
        ),
        (
            buffer(Scalar::F16, Access::Read),
            buffer(Scalar::F32, Access::Read),
        ),
        (
            buffer(Scalar::F16, Access::Read),
            buffer(Scalar::F16, Access::Write),
        ),
        (
            buffer(Scalar::U16, Access::Read),
            buffer(Scalar::F32, Access::Write),
        ),
    ] {
        let error = validate(&product(from, to)).unwrap_err();
        assert!(
            error
                .message
                .contains("invalid SIMD-group matrix operation"),
            "{error}"
        );
    }
}

#[test]
fn matrices_live_in_arrays_but_never_cross_the_abi() {
    let mut types = TypeTable::default();
    let array = types.intern(Aggregate::Array {
        element: Type::Matrix(Scalar::F32),
        length: 4,
    });
    assert!(types.is_owned(array));
    assert!(!types.is_abi_value(array));
    assert!(!types.is_abi_value(Type::Matrix(Scalar::F32)));
    let mut module = kernel(vec![], vec![array], vec![]);
    module.types = types;
    validate(&module).unwrap();
    let parameter = kernel(vec![Type::Matrix(Scalar::F32)], vec![], vec![]);
    let error = validate(&parameter).unwrap_err();
    assert!(
        error.message.contains("unsupported kernel parameter type"),
        "{error}"
    );
    let local = kernel(vec![], vec![Type::Matrix(Scalar::U32)], vec![]);
    let error = validate(&local).unwrap_err();
    assert!(error.message.contains("unsupported local type"), "{error}");
}

fn block(statements: Vec<Statement>, terminator: Terminator) -> Block {
    Block {
        statements,
        terminator,
        source: source(),
    }
}

fn branch(condition: usize, then_block: usize, else_block: usize) -> Terminator {
    Terminator::Branch {
        condition: Operand::local(condition),
        then_block,
        else_block,
    }
}

fn below(local: usize, limit: u32) -> Expression {
    Expression::Binary(BinaryOp::Lt, Operand::local(local), uint(limit))
}

const U32: Type = Type::Scalar(Scalar::U32);
const BOOL: Type = Type::Scalar(Scalar::Bool);
const MATRIX: Type = Type::Matrix(Scalar::F32);

fn output() -> Type {
    buffer(Scalar::F32, Access::Write)
}

fn splat() -> Expression {
    matrix(MatrixOp::Splat, vec![Operand::Constant(Constant::F32(0))])
}

fn store(row: Operand, value: usize) -> Statement {
    assign(
        0,
        matrix(
            MatrixOp::Store { transposed: false },
            vec![
                Operand::local(1),
                row,
                uint(0),
                uint(8),
                Operand::local(value),
            ],
        ),
    )
}

/// A kernel with `blocks` over locals `[unit, output, coordinate: u32, bool, matrix, u32]`.
fn branching(blocks: Vec<Block>) -> Module {
    Module {
        types: TypeTable::default(),
        functions: vec![Function {
            name: "kernel".into(),
            kernel: true,
            required_block: Some([64, 1, 1]),
            requires: Vec::new(),
            parameters: 1,
            locals: vec![Type::Unit, output(), U32, BOOL, MATRIX, U32],
            blocks,
            source: source(),
        }],
    }
}

/// Splats and stores a matrix under a branch on the SIMD coordinate `builtin`.
fn branch_on(builtin: SimdBuiltin) -> Module {
    branching(vec![
        block(
            vec![
                assign(2, Expression::SimdCoordinate(builtin)),
                assign(3, below(2, 1)),
            ],
            branch(3, 1, 2),
        ),
        block(
            vec![assign(4, splat()), store(uint(0), 4)],
            Terminator::Goto(2),
        ),
        block(vec![], Terminator::Return),
    ])
}

#[test]
fn matrix_operations_follow_group_uniform_branches() {
    validate(&branch_on(SimdBuiltin::Group)).unwrap();
    let error = validate(&branch_on(SimdBuiltin::Lane)).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}

/// Stores a splat at a row computed from the SIMD coordinate `builtin`.
fn store_at(builtin: SimdBuiltin) -> Module {
    branching(vec![block(
        vec![
            assign(2, Expression::SimdCoordinate(builtin)),
            assign(
                5,
                Expression::Binary(BinaryOp::Mul, Operand::local(2), uint(8)),
            ),
            assign(4, splat()),
            store(Operand::local(5), 4),
        ],
        Terminator::Return,
    )])
}

#[test]
fn matrix_operands_must_be_uniform_within_the_simd_group() {
    validate(&store_at(SimdBuiltin::Group)).unwrap();
    let error = validate(&store_at(SimdBuiltin::Lane)).unwrap_err();
    assert!(
        error.message.contains("uniform within the SIMD group"),
        "{error}"
    );
    assert_eq!(error.source, source());
}

#[test]
fn a_matrix_assigned_under_lane_control_is_lane_dependent() {
    // Local 4 takes local 6's matrix only on some lanes; storing it then mixes two matrices.
    let mut module = branching(vec![
        block(
            vec![
                assign(2, Expression::SimdCoordinate(SimdBuiltin::Lane)),
                assign(3, below(2, 16)),
                assign(4, splat()),
                assign(6, splat()),
            ],
            branch(3, 1, 2),
        ),
        block(
            vec![assign(4, Expression::Use(Operand::local(6)))],
            Terminator::Goto(2),
        ),
        block(vec![store(uint(0), 4)], Terminator::Return),
    ]);
    module.functions[0].locals.push(MATRIX);
    let error = validate(&module).unwrap_err();
    assert!(
        error.message.contains("uniform within the SIMD group"),
        "{error}"
    );
}

#[test]
fn writing_a_matrix_at_a_lane_dependent_index_makes_it_lane_dependent() {
    let mut types = TypeTable::default();
    let pair = types.intern(Aggregate::Array {
        element: MATRIX,
        length: 2,
    });
    let at = |index: usize| Place {
        local: 6,
        projection: vec![Projection::Index(index)],
    };
    let usize = Type::Scalar(Scalar::Usize);
    // Locals: unit, output, lane, lane as usize, matrix, lane & 1, pair, zero.
    let module = Module {
        types,
        functions: vec![Function {
            name: "kernel".into(),
            kernel: true,
            required_block: Some([64, 1, 1]),
            requires: Vec::new(),
            parameters: 1,
            locals: vec![Type::Unit, output(), U32, usize, MATRIX, usize, pair, usize],
            blocks: vec![block(
                vec![
                    assign(2, Expression::SimdCoordinate(SimdBuiltin::Lane)),
                    assign(3, Expression::Cast(Operand::local(2), Scalar::Usize)),
                    assign(
                        5,
                        Expression::Binary(
                            BinaryOp::BitAnd,
                            Operand::local(3),
                            Operand::Constant(Constant::Usize(1)),
                        ),
                    ),
                    assign(7, Expression::Use(Operand::Constant(Constant::Usize(0)))),
                    assign(4, splat()),
                    assign(
                        6,
                        Expression::Aggregate {
                            ty: 0,
                            fields: vec![Operand::local(4), Operand::local(4)],
                        },
                    ),
                    Statement {
                        destination: at(5),
                        value: Expression::Use(Operand::local(4)),
                        source: source(),
                    },
                    assign(4, Expression::Use(Operand::Place(at(7)))),
                    store(uint(0), 4),
                ],
                Terminator::Return,
            )],
            source: source(),
        }],
    };
    let error = validate(&module).unwrap_err();
    assert!(
        error.message.contains("uniform within the SIMD group"),
        "{error}"
    );
}

#[test]
fn a_lane_dependent_loop_exit_precedes_later_matrix_operations() {
    // while true { if lane < 16 { break } store }: the store is control-dependent on the exit test.
    let module = branching(vec![
        block(
            vec![assign(2, Expression::SimdCoordinate(SimdBuiltin::Lane))],
            Terminator::Goto(1),
        ),
        block(vec![assign(3, below(2, 16))], branch(3, 3, 2)),
        block(
            vec![assign(4, splat()), store(uint(0), 4)],
            Terminator::Goto(1),
        ),
        block(vec![], Terminator::Return),
    ]);
    let error = validate(&module).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}

/// A helper storing a splat at its row parameter, called with the SIMD coordinate `builtin`.
fn helper_call(builtin: SimdBuiltin, required_block: Option<[u32; 3]>) -> Module {
    let helper = Function {
        name: "helper".into(),
        kernel: false,
        required_block: None,
        requires: Vec::new(),
        parameters: 2,
        locals: vec![Type::Unit, output(), U32, MATRIX],
        blocks: vec![block(
            vec![assign(3, splat()), store(Operand::local(2), 3)],
            Terminator::Return,
        )],
        source: source(),
    };
    let kernel = Function {
        name: "kernel".into(),
        kernel: true,
        required_block,
        requires: Vec::new(),
        parameters: 1,
        locals: vec![Type::Unit, output(), U32],
        blocks: vec![block(
            vec![
                assign(2, Expression::SimdCoordinate(builtin)),
                assign(
                    0,
                    Expression::Call {
                        function: 0,
                        arguments: vec![Operand::local(1), Operand::local(2)],
                    },
                ),
            ],
            Terminator::Return,
        )],
        source: source(),
    };
    Module {
        types: TypeTable::default(),
        functions: vec![helper, kernel],
    }
}

#[test]
fn helpers_pass_matrix_participation_to_their_callers() {
    validate(&helper_call(SimdBuiltin::Group, Some([64, 1, 1]))).unwrap();
    let error = validate(&helper_call(SimdBuiltin::Lane, Some([64, 1, 1]))).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}

#[test]
fn matrix_kernels_need_a_literal_block_of_whole_simd_groups() {
    for shape in [None, Some([48, 1, 1]), Some([16, 1, 1])] {
        let error = validate(&helper_call(SimdBuiltin::Group, shape)).unwrap_err();
        assert!(error.message.contains("multiple of 32"), "{error}");
    }
    validate(&helper_call(SimdBuiltin::Group, Some([16, 2, 1]))).unwrap();
}

/// Asserts `coordinate < 1` for the SIMD coordinate `builtin`, then runs `after`.
fn checked(builtin: SimdBuiltin, after: Vec<Statement>) -> Module {
    branching(vec![
        block(
            vec![
                assign(2, Expression::SimdCoordinate(builtin)),
                assign(3, below(2, 1)),
            ],
            Terminator::Assert {
                condition: Operand::local(3),
                expected: true,
                enabled: true,
                target: 1,
                check: Check::Overflow(OverflowOp::Add),
            },
        ),
        block(after, Terminator::Return),
    ])
}

#[test]
fn group_uniform_checks_may_precede_matrix_operations_but_not_barriers() {
    let matrices = || vec![assign(4, splat()), store(uint(0), 4)];
    validate(&checked(SimdBuiltin::Group, matrices())).unwrap();
    let error = validate(&checked(SimdBuiltin::Lane, matrices())).unwrap_err();
    assert!(
        error.message.contains("must be proved or uniform"),
        "{error}"
    );
    let mut barrier = matrices();
    barrier.push(assign(0, Expression::ThreadgroupBarrier));
    let error = validate(&checked(SimdBuiltin::Group, barrier)).unwrap_err();
    assert!(
        error.message.contains("must be proved or uniform"),
        "{error}"
    );
}

#[test]
fn a_group_uniform_check_in_a_helper_before_a_caller_barrier_is_rejected() {
    let helper = Function {
        name: "helper".into(),
        kernel: false,
        required_block: None,
        requires: Vec::new(),
        parameters: 1,
        locals: vec![Type::Unit, U32, BOOL],
        blocks: vec![
            block(
                vec![assign(2, below(1, 1))],
                Terminator::Assert {
                    condition: Operand::local(2),
                    expected: true,
                    enabled: true,
                    target: 1,
                    check: Check::Overflow(OverflowOp::Add),
                },
            ),
            block(vec![], Terminator::Return),
        ],
        source: source(),
    };
    let kernel = |barrier: bool| Function {
        name: "kernel".into(),
        kernel: true,
        required_block: Some([64, 1, 1]),
        requires: Vec::new(),
        parameters: 0,
        locals: vec![Type::Unit, U32],
        blocks: vec![block(
            [
                vec![
                    assign(1, Expression::SimdCoordinate(SimdBuiltin::Group)),
                    assign(
                        0,
                        Expression::Call {
                            function: 0,
                            arguments: vec![Operand::local(1)],
                        },
                    ),
                ],
                if barrier {
                    vec![assign(0, Expression::ThreadgroupBarrier)]
                } else {
                    vec![]
                },
            ]
            .concat(),
            Terminator::Return,
        )],
        source: source(),
    };
    let module = |barrier| Module {
        types: TypeTable::default(),
        functions: vec![helper.clone(), kernel(barrier)],
    };
    validate(&module(false)).unwrap();
    let error = validate(&module(true)).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}
