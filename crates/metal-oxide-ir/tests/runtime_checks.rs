use metal_oxide_ir::*;

fn source() -> SourceLocation {
    SourceLocation {
        file: "kernel.rs".into(),
        line: 4,
        column: 9,
    }
}

fn assign(local: usize, value: Expression) -> Statement {
    Statement {
        destination: Place::local(local),
        value,
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

fn function(kernel: bool, parameters: usize, locals: Vec<Type>, blocks: Vec<Block>) -> Function {
    Function {
        name: if kernel { "kernel" } else { "helper" }.into(),
        kernel,
        required_block: None,
        parameters,
        locals,
        blocks,
        source: source(),
    }
}

fn module(functions: Vec<Function>) -> Module {
    Module {
        types: TypeTable::default(),
        functions,
    }
}

fn below(local: usize, limit: u32) -> Expression {
    Expression::Binary(
        BinaryOp::Lt,
        Operand::local(local),
        Operand::Constant(Constant::U32(limit)),
    )
}

fn assertion(condition: usize, target: usize) -> Terminator {
    Terminator::Assert {
        condition: Operand::local(condition),
        expected: true,
        enabled: true,
        target,
        check: Check::Overflow(OverflowOp::Add),
    }
}

const U32: Type = Type::Scalar(Scalar::U32);
const BOOL: Type = Type::Scalar(Scalar::Bool);

#[test]
fn a_division_guarded_by_runtime_checks_needs_no_range_proof() {
    // `x / y` as rustc emits it: a division-by-zero assertion before the division.
    let divide = function(
        false,
        2,
        vec![U32, U32, U32, BOOL],
        vec![
            block(
                vec![assign(
                    3,
                    Expression::Binary(
                        BinaryOp::Eq,
                        Operand::local(2),
                        Operand::Constant(Constant::U32(0)),
                    ),
                )],
                Terminator::Assert {
                    condition: Operand::local(3),
                    expected: false,
                    enabled: true,
                    target: 1,
                    check: Check::DivisionByZero,
                },
            ),
            block(
                vec![assign(
                    0,
                    Expression::Binary(BinaryOp::Div, Operand::local(1), Operand::local(2)),
                )],
                Terminator::Return,
            ),
        ],
    );
    let analyses = validate(&module(vec![divide.clone()])).unwrap();
    assert_eq!(analyses[0].checks.iter().copied().collect::<Vec<_>>(), [0]);
    assert!(analyses[0].fallible);
    let mut unguarded = divide;
    unguarded.blocks[0].terminator = Terminator::Goto(1);
    let error = validate(&module(vec![unguarded])).unwrap_err();
    assert!(error.message.contains("nonzero divisor"), "{error}");
}

#[test]
fn proven_assertions_and_unreachable_panics_emit_no_runtime_check() {
    let guarded = |constant: Option<u32>| {
        let mut statements = vec![assign(2, below(1, 4))];
        if let Some(value) = constant {
            statements.insert(
                0,
                assign(1, Expression::Use(Operand::Constant(Constant::U32(value)))),
            );
        }
        function(
            false,
            if constant.is_some() { 0 } else { 1 },
            vec![Type::Unit, U32, BOOL],
            vec![
                block(
                    statements,
                    Terminator::Branch {
                        condition: Operand::local(2),
                        then_block: 1,
                        else_block: 2,
                    },
                ),
                block(vec![], assertion(2, 3)),
                block(vec![], Terminator::Panic("too large".into())),
                block(vec![], Terminator::Return),
            ],
        )
    };
    let analyses = validate(&module(vec![guarded(Some(2))])).unwrap();
    assert!(analyses[0].checks.is_empty());
    assert!(!analyses[0].fallible);
    // An unknown argument can reach the panic; the guarded assertion stays proven.
    let analyses = validate(&module(vec![guarded(None)])).unwrap();
    assert_eq!(analyses[0].checks.iter().copied().collect::<Vec<_>>(), [2]);
}

fn barrier() -> Statement {
    assign(0, Expression::ThreadgroupBarrier)
}

/// A kernel that asserts a uniform or varying condition before or after a barrier.
fn kernel(uniform: bool, before_barrier: bool) -> Module {
    let value = if uniform {
        Expression::Use(Operand::local(1))
    } else {
        Expression::SimdCoordinate(SimdBuiltin::Lane)
    };
    let mut blocks = vec![
        block(
            vec![assign(2, value), assign(3, below(2, 4))],
            assertion(3, 1),
        ),
        block(vec![], Terminator::Return),
    ];
    if before_barrier {
        blocks[1].statements.push(barrier());
    } else {
        blocks[0].statements.insert(0, barrier());
    }
    module(vec![function(
        true,
        1,
        vec![Type::Unit, U32, U32, BOOL],
        blocks,
    )])
}

#[test]
fn checks_before_a_barrier_must_be_uniform() {
    for (uniform, before_barrier) in [(true, true), (false, false), (true, false)] {
        let analyses = validate(&kernel(uniform, before_barrier)).unwrap();
        assert_eq!(analyses[0].checks.len(), 1);
    }
    let error = validate(&kernel(false, true)).unwrap_err();
    assert!(
        error.message.contains("must be proved or uniform"),
        "{error}"
    );
    assert_eq!(error.source, source());
}

#[test]
fn a_panic_on_some_threads_before_a_barrier_is_rejected() {
    let module = module(vec![function(
        true,
        0,
        vec![Type::Unit, U32, BOOL],
        vec![
            block(
                vec![
                    assign(1, Expression::SimdCoordinate(SimdBuiltin::Lane)),
                    assign(2, below(1, 4)),
                ],
                Terminator::Branch {
                    condition: Operand::local(2),
                    then_block: 1,
                    else_block: 2,
                },
            ),
            block(vec![barrier()], Terminator::Return),
            block(vec![], Terminator::Panic("lane out of range".into())),
        ],
    )]);
    let error = validate(&module).unwrap_err();
    assert!(
        error.message.contains("must be proved or uniform"),
        "{error}"
    );
}

#[test]
fn helpers_with_checks_follow_their_call_sites() {
    let helper = function(
        false,
        1,
        vec![Type::Unit, U32, BOOL],
        vec![
            block(vec![assign(2, below(1, 4))], assertion(2, 1)),
            block(vec![], Terminator::Return),
        ],
    );
    let call = assign(
        0,
        Expression::Call {
            function: 0,
            arguments: vec![Operand::local(1)],
        },
    );
    let caller = |before_barrier: bool| {
        let lane = assign(1, Expression::SimdCoordinate(SimdBuiltin::Lane));
        let statements = if before_barrier {
            vec![lane, call.clone(), barrier()]
        } else {
            vec![lane, barrier(), call.clone()]
        };
        module(vec![
            helper.clone(),
            function(
                true,
                0,
                vec![Type::Unit, U32],
                vec![block(statements, Terminator::Return)],
            ),
        ])
    };
    let analyses = validate(&caller(false)).unwrap();
    assert!(analyses[0].fallible && analyses[1].fallible);
    assert!(!analyses[0].cooperative);
    let error = validate(&caller(true)).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}

fn axis(local: usize, axis: u32) -> Operand {
    Operand::Place(Place {
        local,
        projection: vec![Projection::Field(axis)],
    })
}

fn overflow(local: usize, op: OverflowOp, target: usize) -> Terminator {
    Terminator::Assert {
        condition: axis(local, 1),
        expected: false,
        enabled: true,
        target,
        check: Check::Overflow(op),
    }
}

/// `a.x * b.axis + thread_idx.x` with rustc's overflow assertions; helpers read coordinates.
fn launch_index(a: Builtin, b: (Builtin, u32)) -> Vec<usize> {
    let coordinates = |builtin| {
        function(
            false,
            0,
            vec![Type::Dim3],
            vec![block(
                vec![assign(0, Expression::Coordinates(builtin))],
                Terminator::Return,
            )],
        )
    };
    let call = |local, function| {
        assign(
            local,
            Expression::Call {
                function,
                arguments: vec![],
            },
        )
    };
    let checked = |local, op, x, y| assign(local, Expression::Binary(op, x, y));
    let kernel = function(
        true,
        0,
        vec![
            Type::Unit,
            Type::Dim3,
            Type::Dim3,
            Type::Dim3,
            Type::Checked(Scalar::U32),
            Type::Checked(Scalar::U32),
        ],
        vec![
            block(
                vec![
                    call(1, 0),
                    call(2, 1),
                    call(3, 2),
                    checked(4, BinaryOp::MulWithOverflow, axis(1, 0), axis(2, b.1)),
                ],
                overflow(4, OverflowOp::Mul, 1),
            ),
            block(
                vec![checked(
                    5,
                    BinaryOp::AddWithOverflow,
                    axis(4, 0),
                    axis(3, 0),
                )],
                overflow(5, OverflowOp::Add, 2),
            ),
            block(vec![], Terminator::Return),
        ],
    );
    let analyses = validate(&module(vec![
        coordinates(a),
        coordinates(b.0),
        coordinates(Builtin::ThreadIdx),
        kernel,
    ]))
    .unwrap();
    analyses[3].checks.iter().copied().collect()
}

#[test]
fn launch_coordinates_prove_global_index_arithmetic() {
    assert_eq!(launch_index(Builtin::BlockIdx, (Builtin::BlockDim, 0)), []);
    assert_eq!(launch_index(Builtin::BlockDim, (Builtin::BlockIdx, 0)), []);
    assert_eq!(launch_index(Builtin::GridDim, (Builtin::BlockDim, 0)), [1]);
    assert_eq!(
        launch_index(Builtin::BlockIdx, (Builtin::BlockDim, 1)),
        [0, 1]
    );
    assert_eq!(
        launch_index(Builtin::BlockIdx, (Builtin::BlockIdx, 0)),
        [0, 1]
    );
}

/// `block_idx.x * 64 + thread_idx.x` in a kernel that may require a block size.
fn constant_block_index(required_block: Option<[u32; 3]>) -> Vec<usize> {
    let mut kernel = function(
        true,
        0,
        vec![
            Type::Unit,
            Type::Dim3,
            Type::Dim3,
            Type::Checked(Scalar::U32),
            Type::Checked(Scalar::U32),
        ],
        vec![
            block(
                vec![
                    assign(1, Expression::Coordinates(Builtin::BlockIdx)),
                    assign(2, Expression::Coordinates(Builtin::ThreadIdx)),
                    assign(
                        3,
                        Expression::Binary(
                            BinaryOp::MulWithOverflow,
                            axis(1, 0),
                            Operand::Constant(Constant::U32(64)),
                        ),
                    ),
                ],
                overflow(3, OverflowOp::Mul, 1),
            ),
            block(
                vec![assign(
                    4,
                    Expression::Binary(BinaryOp::AddWithOverflow, axis(3, 0), axis(2, 0)),
                )],
                overflow(4, OverflowOp::Add, 2),
            ),
            block(vec![], Terminator::Return),
        ],
    );
    kernel.required_block = required_block;
    let analyses = validate(&module(vec![kernel])).unwrap();
    analyses[0].checks.iter().copied().collect()
}

#[test]
fn a_required_block_bounds_the_constant_block_index() {
    assert_eq!(constant_block_index(Some([64, 1, 1])), []);
    assert_eq!(constant_block_index(Some([128, 1, 1])), []);
    // A stride larger than the block can overflow on the last blocks of a large grid.
    assert_eq!(constant_block_index(Some([32, 1, 1])), [0, 1]);
    assert_eq!(constant_block_index(None), [0, 1]);
}

/// `thread_index * factor` with rustc's overflow assertion; a helper reads the index.
fn scaled_thread_index(factor: u64) -> (Vec<usize>, Vec<bool>) {
    let index = function(
        false,
        0,
        vec![Type::Scalar(Scalar::Usize)],
        vec![block(
            vec![assign(0, Expression::ThreadIndex)],
            Terminator::Return,
        )],
    );
    let kernel = function(
        true,
        0,
        vec![
            Type::Unit,
            Type::Scalar(Scalar::Usize),
            Type::Checked(Scalar::Usize),
        ],
        vec![
            block(
                vec![
                    assign(
                        1,
                        Expression::Call {
                            function: 0,
                            arguments: vec![],
                        },
                    ),
                    assign(
                        2,
                        Expression::Binary(
                            BinaryOp::MulWithOverflow,
                            Operand::local(1),
                            Operand::Constant(Constant::Usize(factor)),
                        ),
                    ),
                ],
                overflow(2, OverflowOp::Mul, 1),
            ),
            block(vec![], Terminator::Return),
        ],
    );
    let analyses = validate(&module(vec![index, kernel])).unwrap();
    (
        analyses[1].checks.iter().copied().collect(),
        analyses.iter().map(|a| a.one_dimensional).collect(),
    )
}

#[test]
fn the_thread_index_stays_below_u32_max_in_a_one_dimensional_launch() {
    assert_eq!(scaled_thread_index(1 << 32), (vec![], vec![true, true]));
    assert_eq!(scaled_thread_index(1 << 33).0, [0]);
}

/// `((thread_idx.x >> 24) << shift) + addend` with rustc's overflow assertion.
fn shifted_index(shift: u32, addend: u32) -> Vec<usize> {
    let constant = |value| Operand::Constant(Constant::U32(value));
    let kernel = function(
        true,
        0,
        vec![Type::Unit, Type::Dim3, U32, U32, Type::Checked(Scalar::U32)],
        vec![
            block(
                vec![
                    assign(1, Expression::Coordinates(Builtin::ThreadIdx)),
                    assign(
                        2,
                        Expression::Binary(BinaryOp::Shr, axis(1, 0), constant(24)),
                    ),
                    assign(
                        3,
                        Expression::Binary(BinaryOp::Shl, Operand::local(2), constant(shift)),
                    ),
                    assign(
                        4,
                        Expression::Binary(
                            BinaryOp::AddWithOverflow,
                            Operand::local(3),
                            constant(addend),
                        ),
                    ),
                ],
                overflow(4, OverflowOp::Add, 1),
            ),
            block(vec![], Terminator::Return),
        ],
    );
    let analyses = validate(&module(vec![kernel])).unwrap();
    analyses[0].checks.iter().copied().collect()
}

#[test]
fn shifts_of_bounded_values_bound_later_arithmetic() {
    assert_eq!(shifted_index(4, u32::MAX - 4080), []);
    assert_eq!(shifted_index(4, u32::MAX - 4079), [0]);
    assert_eq!(shifted_index(24, 0x00ff_ffff), []);
    assert_eq!(shifted_index(25, 1), [0]);
}
