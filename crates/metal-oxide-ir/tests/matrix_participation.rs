mod support;

use metal_oxide_ir::*;
use support::*;

/// `if coordinate < 1 { then }` over the SIMD coordinate `builtin`.
fn under_branch(builtin: SimdBuiltin, then: Vec<Statement>) -> Function {
    kernel(vec![
        block(
            vec![
                assign(2, Expression::SimdCoordinate(builtin)),
                assign(3, below(2, 1)),
            ],
            branch(3, 1, 2),
        ),
        block(then, Terminator::Goto(2)),
        block(vec![], Terminator::Return),
    ])
}

#[test]
fn matrix_operations_follow_group_uniform_branches() {
    let branch_on = |builtin| module(vec![under_branch(builtin, matrices())]);
    validate(&branch_on(SimdBuiltin::Group)).unwrap();
    let error = validate(&branch_on(SimdBuiltin::Lane)).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}

/// Stores a splat at a row computed from the SIMD coordinate `builtin`.
fn store_at(builtin: SimdBuiltin) -> Module {
    module(vec![kernel(vec![block(
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
    )])])
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
    let mut mixed = kernel(vec![
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
    mixed.locals.push(MATRIX);
    let error = validate(&module(vec![mixed])).unwrap_err();
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
    let locals = vec![Type::Unit, output(), U32, usize, MATRIX, usize, pair, usize];
    let statements = vec![
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
    ];
    let module = Module {
        types,
        functions: vec![function(
            "kernel",
            true,
            locals,
            vec![block(statements, Terminator::Return)],
        )],
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
    let exit = kernel(vec![
        block(
            vec![assign(2, Expression::SimdCoordinate(SimdBuiltin::Lane))],
            Terminator::Goto(1),
        ),
        block(vec![assign(3, below(2, 16))], branch(3, 3, 2)),
        block(matrices(), Terminator::Goto(1)),
        block(vec![], Terminator::Return),
    ]);
    let error = validate(&module(vec![exit])).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}

/// `loop { matrices; if coordinate < 16 { break } }` over the SIMD coordinate `builtin`.
fn exit_after_matrices(builtin: SimdBuiltin) -> Module {
    module(vec![kernel(vec![
        block(
            vec![assign(2, Expression::SimdCoordinate(builtin))],
            Terminator::Goto(1),
        ),
        block(matrices(), Terminator::Goto(2)),
        block(vec![assign(3, below(2, 16))], branch(3, 3, 1)),
        block(vec![], Terminator::Return),
    ])])
}

#[test]
fn loop_carried_lane_control_reaches_matrix_operations_before_the_exit_test() {
    validate(&exit_after_matrices(SimdBuiltin::Group)).unwrap();
    let error = validate(&exit_after_matrices(SimdBuiltin::Lane)).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}

#[test]
fn group_uniform_control_never_admits_a_barrier() {
    let barrier_under_branch = |builtin| module(vec![under_branch(builtin, vec![barrier()])]);
    validate(&barrier_under_branch(SimdBuiltin::Count)).unwrap();
    for builtin in [SimdBuiltin::Group, SimdBuiltin::Lane] {
        let error = validate(&barrier_under_branch(builtin)).unwrap_err();
        assert!(error.message.contains("uniform participation"), "{error}");
    }
}

/// `shuffle_xor(1, coordinate)` over the SIMD coordinate `builtin`.
fn shuffle_control(builtin: SimdBuiltin) -> Module {
    let shuffle = Expression::Simd {
        op: SimdOp::ShuffleXor,
        arguments: vec![uint(1), Operand::local(2)],
    };
    module(vec![kernel(vec![block(
        vec![
            assign(2, Expression::SimdCoordinate(builtin)),
            assign(5, shuffle),
        ],
        Terminator::Return,
    )])])
}

#[test]
fn group_uniform_shuffle_controls_are_rejected() {
    validate(&shuffle_control(SimdBuiltin::Count)).unwrap();
    for builtin in [SimdBuiltin::Group, SimdBuiltin::Lane] {
        let error = validate(&shuffle_control(builtin)).unwrap_err();
        assert!(error.message.contains("uniform participation"), "{error}");
    }
}

/// `helper(output, row)` stores a splat at `row`.
fn store_helper() -> Function {
    let mut helper = function(
        "helper",
        false,
        vec![Type::Unit, output(), U32, MATRIX],
        vec![block(
            vec![assign(3, splat()), store(Operand::local(2), 3)],
            Terminator::Return,
        )],
    );
    helper.parameters = 2;
    helper
}

/// Calls [`store_helper`] with the kernel's output and `row`.
fn store_call(row: Operand) -> Statement {
    assign(
        0,
        Expression::Call {
            function: 0,
            arguments: vec![Operand::local(1), row],
        },
    )
}

/// Calls [`store_helper`] at a row of the SIMD coordinate `builtin`.
fn helper_call(builtin: SimdBuiltin, required_block: Option<[u32; 3]>) -> Module {
    let mut kernel = function(
        "kernel",
        true,
        vec![Type::Unit, output(), U32],
        vec![block(
            vec![
                assign(2, Expression::SimdCoordinate(builtin)),
                store_call(Operand::local(2)),
            ],
            Terminator::Return,
        )],
    );
    kernel.required_block = required_block;
    module(vec![store_helper(), kernel])
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

#[test]
fn a_matrix_helper_called_under_a_branch_needs_group_uniform_control() {
    let called = |builtin| {
        module(vec![
            store_helper(),
            under_branch(builtin, vec![store_call(uint(0))]),
        ])
    };
    validate(&called(SimdBuiltin::Group)).unwrap();
    let error = validate(&called(SimdBuiltin::Lane)).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}
