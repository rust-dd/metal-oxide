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

/// `name(q) { f(q) }` over `f` as function 0, returning from a block after the call's.
fn forward(name: &str) -> Function {
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
        forward("a"),
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
    module(vec![f(), forward("m"), uniform, lane])
}

#[test]
fn a_check_in_a_shared_helper_constrains_only_calls_that_precede_cooperation() {
    validate(&shared_helper(vec![])).unwrap();
    let error = validate(&shared_helper(vec![barrier()])).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}

/// `loop { barrier; f(local) }` in a kernel over `[unit, n, lane]`.
fn barrier_loop(local: usize) -> Module {
    let looping = function(
        "kernel",
        true,
        vec![Type::Unit, U32, U32],
        vec![
            block(
                vec![assign(2, Expression::SimdCoordinate(SimdBuiltin::Lane))],
                Terminator::Goto(1),
            ),
            block(vec![barrier(), call(0, local)], Terminator::Goto(1)),
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
