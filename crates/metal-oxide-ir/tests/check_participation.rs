mod support;

use metal_oxide_ir::*;
use support::*;

/// Asserts that local `condition` holds, then continues at block 1.
fn assertion(condition: usize) -> Terminator {
    Terminator::Assert {
        condition: Operand::local(condition),
        expected: true,
        enabled: true,
        target: 1,
        check: Check::Overflow(OverflowOp::Add),
    }
}

/// `f(p) { assert p < 1 }`.
fn f() -> Function {
    function(
        "f",
        false,
        vec![Type::Unit, U32, BOOL],
        vec![
            block(vec![assign(2, below(1, 1))], assertion(2)),
            block(vec![], Terminator::Return),
        ],
    )
}

/// `name(q) { f(q) }` over `f` as function 0; the call's block jumps to a block that returns.
fn forward_then_goto(name: &str) -> Function {
    function(
        name,
        false,
        vec![Type::Unit, U32],
        vec![
            block(vec![call(0, 1)], Terminator::Goto(1)),
            block(vec![], Terminator::Return),
        ],
    )
}

/// `kernel(output) { a(coordinate); after }`, `a(q) { f(q) }` and `f(p) { assert p < 1 }`.
fn nested_check(builtin: SimdBuiltin, after: Vec<Statement>) -> Module {
    let statements = [
        vec![assign(2, Expression::SimdCoordinate(builtin)), call(1, 2)],
        after,
    ];
    module(vec![
        f(),
        forward_then_goto("a"),
        kernel(vec![block(statements.concat(), Terminator::Return)]),
    ])
}

#[test]
fn a_check_behind_nested_helpers_precedes_later_matrix_operations() {
    let error = validate(&nested_check(SimdBuiltin::Lane, matrices())).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
    let analyses = validate(&nested_check(SimdBuiltin::Group, matrices())).unwrap();
    assert!(!analyses[1].cooperative, "the helper that calls the check");
}

#[test]
fn a_check_behind_nested_helpers_precedes_later_barriers() {
    for builtin in [SimdBuiltin::Lane, SimdBuiltin::Group] {
        let error = validate(&nested_check(builtin, vec![barrier()])).unwrap_err();
        assert!(error.message.contains("uniform participation"), "{error}");
    }
}

/// [`nested_check`] with `a` returning from the block of its call.
fn returning_from_the_call(builtin: SimdBuiltin, after: Vec<Statement>) -> Module {
    let mut nested = nested_check(builtin, after);
    nested.functions[1].blocks = vec![block(vec![call(0, 1)], Terminator::Return)];
    nested
}

#[test]
fn a_check_behind_a_call_that_ends_its_helper_precedes_later_cooperation() {
    for after in [matrices(), vec![barrier()]] {
        let error = validate(&returning_from_the_call(SimdBuiltin::Lane, after)).unwrap_err();
        assert!(error.message.contains("uniform participation"), "{error}");
    }
    validate(&returning_from_the_call(SimdBuiltin::Group, matrices())).unwrap();
    let synchronized = returning_from_the_call(SimdBuiltin::Group, vec![barrier()]);
    let error = validate(&synchronized).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}

/// Kernels `uniform(n) { m(n); barrier }` and `lane(n) { m(lane); later }`, `m(q) { f(q) }`.
fn shared_helper(later: Vec<Statement>) -> Module {
    let uniform = function(
        "uniform",
        true,
        vec![Type::Unit, U32],
        vec![block(vec![call(1, 1), barrier()], Terminator::Return)],
    );
    let statements = [
        vec![
            assign(2, Expression::SimdCoordinate(SimdBuiltin::Lane)),
            call(1, 2),
        ],
        later,
    ];
    let lane = function(
        "lane",
        true,
        vec![Type::Unit, U32, U32],
        vec![block(statements.concat(), Terminator::Return)],
    );
    module(vec![f(), forward_then_goto("m"), uniform, lane])
}

#[test]
fn a_check_in_a_shared_helper_constrains_only_calls_that_precede_cooperation() {
    validate(&shared_helper(vec![])).unwrap();
    let error = validate(&shared_helper(vec![barrier()])).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}

/// `if local < 16 { f(n) }; barrier` in a kernel over the locals `[unit, n, lane, more]`.
fn conditional_call(local: usize) -> Module {
    let conditional = function(
        "kernel",
        true,
        vec![Type::Unit, U32, U32, BOOL],
        vec![
            block(
                vec![
                    assign(2, Expression::SimdCoordinate(SimdBuiltin::Lane)),
                    assign(3, below(local, 16)),
                ],
                branch(3, 1, 2),
            ),
            block(vec![call(0, 1)], Terminator::Goto(2)),
            block(vec![barrier()], Terminator::Return),
        ],
    );
    module(vec![f(), conditional])
}

#[test]
fn a_fallible_call_under_lane_control_precedes_later_cooperation() {
    let error = validate(&conditional_call(2)).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
    validate(&conditional_call(1)).unwrap();
}

/// `i = 0; loop { barrier; f(local); i += 1; if i >= n { break } }` in a kernel over the locals
/// `[unit, n, lane, i, more]`.
fn barrier_loop(local: usize) -> Module {
    let next = Expression::Binary(BinaryOp::Add, Operand::local(3), uint(1));
    let more = Expression::Binary(BinaryOp::Lt, Operand::local(3), Operand::local(1));
    let looping = function(
        "kernel",
        true,
        vec![Type::Unit, U32, U32, U32, BOOL],
        vec![
            block(
                vec![
                    assign(2, Expression::SimdCoordinate(SimdBuiltin::Lane)),
                    assign(3, Expression::Use(uint(0))),
                ],
                Terminator::Goto(1),
            ),
            block(
                vec![barrier(), call(0, local), assign(3, next), assign(4, more)],
                branch(4, 1, 2),
            ),
            block(vec![], Terminator::Return),
        ],
    );
    module(vec![f(), looping])
}

#[test]
fn a_fallible_call_in_a_loop_precedes_the_barrier_of_the_next_iteration() {
    let error = validate(&barrier_loop(2)).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
    validate(&barrier_loop(1)).unwrap();
}

/// Asserts `coordinate < 1` for the SIMD coordinate `builtin`, then runs `after`.
fn checked(builtin: SimdBuiltin, after: Vec<Statement>) -> Module {
    module(vec![kernel(vec![
        block(
            vec![
                assign(2, Expression::SimdCoordinate(builtin)),
                assign(3, below(2, 1)),
            ],
            assertion(3),
        ),
        block(after, Terminator::Return),
    ])])
}

#[test]
fn group_uniform_checks_may_precede_matrix_operations_but_not_barriers() {
    validate(&checked(SimdBuiltin::Group, matrices())).unwrap();
    let error = validate(&checked(SimdBuiltin::Lane, matrices())).unwrap_err();
    assert!(
        error.message.contains("must be proved or uniform"),
        "{error}"
    );
    let synchronized = [matrices(), vec![barrier()]].concat();
    let error = validate(&checked(SimdBuiltin::Group, synchronized)).unwrap_err();
    assert!(
        error.message.contains("must be proved or uniform"),
        "{error}"
    );
}

#[test]
fn a_group_uniform_check_in_a_helper_before_a_caller_barrier_is_rejected() {
    let caller = |later: Vec<Statement>| {
        let statements = [
            vec![
                assign(1, Expression::SimdCoordinate(SimdBuiltin::Group)),
                call(0, 1),
            ],
            later,
        ];
        Function {
            name: "kernel".into(),
            kernel: true,
            required_block: Some([64, 1, 1]),
            requires: Vec::new(),
            parameters: 0,
            locals: vec![Type::Unit, U32],
            blocks: vec![block(statements.concat(), Terminator::Return)],
            source: source(),
        }
    };
    validate(&module(vec![f(), caller(vec![])])).unwrap();
    let error = validate(&module(vec![f(), caller(vec![barrier()])])).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}

/// The locals of [`calling`]'s kernel that hold its parameter and the lane.
const PARAMETER: usize = 1;
const LANE: usize = 2;

/// `kernel(n) { lane = SimdCoordinate(Lane); callee(arguments); after }` over function 0.
fn calling(callee: Function, arguments: &[usize], after: Vec<Statement>) -> Module {
    let arguments = arguments
        .iter()
        .map(|&local| Operand::local(local))
        .collect();
    let statements = [
        vec![
            assign(LANE, Expression::SimdCoordinate(SimdBuiltin::Lane)),
            assign(
                0,
                Expression::Call {
                    function: 0,
                    arguments,
                },
            ),
        ],
        after,
    ];
    let kernel = function(
        "kernel",
        true,
        vec![Type::Unit, U32, U32],
        vec![block(statements.concat(), Terminator::Return)],
    );
    module(vec![callee, kernel])
}

/// `panics(p) { if p < 16 { panic } }`, or `panics() { if lane < 16 { panic } }` when it reads
/// the lane; with `goto`, a block of its own jumps to the panic.
fn panicking(reads_lane: bool, goto: bool) -> Function {
    let mut condition = vec![assign(2, below(1, 16))];
    if reads_lane {
        let lane = assign(1, Expression::SimdCoordinate(SimdBuiltin::Lane));
        condition.insert(0, lane);
    }
    let panic = block(vec![], Terminator::Panic("lane out of range".into()));
    let blocks = if goto {
        vec![
            block(condition, branch(2, 1, 3)),
            block(vec![], Terminator::Goto(2)),
            panic,
            block(vec![], Terminator::Return),
        ]
    } else {
        vec![
            block(condition, branch(2, 1, 2)),
            panic,
            block(vec![], Terminator::Return),
        ]
    };
    let mut helper = function("panics", false, vec![Type::Unit, U32, BOOL], blocks);
    helper.parameters = usize::from(!reads_lane);
    helper
}

#[test]
fn a_panic_on_some_lanes_of_a_helper_precedes_later_barriers() {
    for goto in [false, true] {
        let reads_lane = calling(panicking(true, goto), &[], vec![barrier()]);
        let error = validate(&reads_lane).unwrap_err();
        assert!(error.message.contains("uniform participation"), "{error}");
        let argument = |local| calling(panicking(false, goto), &[local], vec![barrier()]);
        let error = validate(&argument(LANE)).unwrap_err();
        assert!(error.message.contains("uniform participation"), "{error}");
        validate(&argument(PARAMETER)).unwrap();
    }
}

/// `synchronizes(x) { barrier; assert x < 1 }`.
fn barrier_then_check() -> Function {
    function(
        "synchronizes",
        false,
        vec![Type::Unit, U32, BOOL],
        vec![
            block(vec![barrier(), assign(2, below(1, 1))], assertion(2)),
            block(vec![], Terminator::Return),
        ],
    )
}

#[test]
fn a_cooperative_callee_that_then_checks_precedes_later_cooperation() {
    let after = |local, later| calling(barrier_then_check(), &[local], later);
    let error = validate(&after(LANE, vec![barrier()])).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
    validate(&after(PARAMETER, vec![barrier()])).unwrap();
    validate(&after(LANE, vec![])).unwrap();
}

/// `name(q) { callee(q) }` in a single block that ends with the return.
fn forward_then_return(name: &str, callee: usize) -> Function {
    function(
        name,
        false,
        vec![Type::Unit, U32],
        vec![block(vec![call(callee, 1)], Terminator::Return)],
    )
}

#[test]
fn a_check_behind_three_calls_that_end_their_blocks_precedes_later_cooperation() {
    // `f(p) { assert p < 1 }`, `g(q) { f(q) }` and `h(r) { g(r) }` called as `h(coordinate)`.
    let chain = |builtin| {
        let statements = [
            vec![assign(2, Expression::SimdCoordinate(builtin)), call(2, 2)],
            matrices(),
        ];
        module(vec![
            f(),
            forward_then_return("g", 0),
            forward_then_return("h", 1),
            kernel(vec![block(statements.concat(), Terminator::Return)]),
        ])
    };
    let error = validate(&chain(SimdBuiltin::Lane)).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
    validate(&chain(SimdBuiltin::Group)).unwrap();
}
