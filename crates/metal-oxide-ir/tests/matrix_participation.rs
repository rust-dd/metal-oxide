use metal_oxide_ir::*;

const U32: Type = Type::Scalar(Scalar::U32);

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

fn block(statements: Vec<Statement>, terminator: Terminator) -> Block {
    Block {
        statements,
        terminator,
        source: source(),
    }
}

fn uint(value: u32) -> Operand {
    Operand::Constant(Constant::U32(value))
}

fn output() -> Type {
    Type::Buffer {
        element: Element::Scalar(Scalar::F32),
        access: Access::Write,
        address_space: AddressSpace::Device,
    }
}

fn below(local: usize, limit: u32) -> Expression {
    Expression::Binary(BinaryOp::Lt, Operand::local(local), uint(limit))
}

fn branch(condition: usize, then_block: usize, else_block: usize) -> Terminator {
    Terminator::Branch {
        condition: Operand::local(condition),
        then_block,
        else_block,
    }
}

fn call(function: usize, argument: usize) -> Statement {
    assign(
        0,
        Expression::Call {
            function,
            arguments: vec![Operand::local(argument)],
        },
    )
}

/// A function of one parameter; a kernel runs in blocks of 64 threads.
fn function(name: &str, kernel: bool, locals: Vec<Type>, blocks: Vec<Block>) -> Function {
    Function {
        name: name.into(),
        kernel,
        required_block: kernel.then_some([64, 1, 1]),
        requires: Vec::new(),
        parameters: 1,
        locals,
        blocks,
        source: source(),
    }
}

/// `kernel(output) { a(coordinate); after }`, `a(q) { f(q) }` and `f(p) { assert p < 1 }`.
fn nested_check(builtin: SimdBuiltin, after: Vec<Statement>) -> Module {
    let assertion = Terminator::Assert {
        condition: Operand::local(2),
        expected: true,
        enabled: true,
        target: 1,
        check: Check::Overflow(OverflowOp::Add),
    };
    let f = vec![
        block(vec![assign(2, below(1, 1))], assertion),
        block(vec![], Terminator::Return),
    ];
    let a = vec![
        block(vec![call(0, 1)], Terminator::Goto(1)),
        block(vec![], Terminator::Return),
    ];
    let kernel = [
        vec![assign(2, Expression::SimdCoordinate(builtin)), call(1, 2)],
        after,
    ];
    Module {
        types: TypeTable::default(),
        functions: vec![
            function(
                "f",
                false,
                vec![Type::Unit, U32, Type::Scalar(Scalar::Bool)],
                f,
            ),
            function("a", false, vec![Type::Unit, U32], a),
            function(
                "kernel",
                true,
                vec![Type::Unit, output(), U32, Type::Matrix(Scalar::F32)],
                vec![block(kernel.concat(), Terminator::Return)],
            ),
        ],
    }
}

/// Splats a matrix and stores it to the kernel's output.
fn matrices() -> Vec<Statement> {
    let splat = Expression::Matrix {
        op: MatrixOp::Splat,
        arguments: vec![Operand::Constant(Constant::F32(0))],
    };
    let store = Expression::Matrix {
        op: MatrixOp::Store { transposed: false },
        arguments: vec![
            Operand::local(1),
            uint(0),
            uint(0),
            uint(8),
            Operand::local(3),
        ],
    };
    vec![assign(3, splat), assign(0, store)]
}

#[test]
fn a_check_behind_nested_helpers_precedes_later_matrix_operations() {
    let error = validate(&nested_check(SimdBuiltin::Lane, matrices())).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
    let analyses = validate(&nested_check(SimdBuiltin::Group, matrices())).unwrap();
    assert!(analyses[1].cooperative, "the helper that calls the check");
}

#[test]
fn a_check_behind_nested_helpers_precedes_later_barriers() {
    let barrier = || vec![assign(0, Expression::ThreadgroupBarrier)];
    for builtin in [SimdBuiltin::Lane, SimdBuiltin::Group] {
        let error = validate(&nested_check(builtin, barrier())).unwrap_err();
        assert!(error.message.contains("uniform participation"), "{error}");
    }
}

/// A kernel over the locals `[unit, output, coordinate, matrix, condition, shuffled]`.
fn kernel_of(blocks: Vec<Block>) -> Module {
    let locals = vec![
        Type::Unit,
        output(),
        U32,
        Type::Matrix(Scalar::F32),
        Type::Scalar(Scalar::Bool),
        U32,
    ];
    Module {
        types: TypeTable::default(),
        functions: vec![function("kernel", true, locals, blocks)],
    }
}

/// `loop { matrices; if coordinate < 16 { break } }` over the SIMD coordinate `builtin`.
fn exit_after_matrices(builtin: SimdBuiltin) -> Module {
    kernel_of(vec![
        block(
            vec![assign(2, Expression::SimdCoordinate(builtin))],
            Terminator::Goto(1),
        ),
        block(matrices(), Terminator::Goto(2)),
        block(vec![assign(4, below(2, 16))], branch(4, 3, 1)),
        block(vec![], Terminator::Return),
    ])
}

/// `if coordinate < 1 { barrier }` over the SIMD coordinate `builtin`.
fn barrier_under_branch(builtin: SimdBuiltin) -> Module {
    kernel_of(vec![
        block(
            vec![
                assign(2, Expression::SimdCoordinate(builtin)),
                assign(4, below(2, 1)),
            ],
            branch(4, 1, 2),
        ),
        block(
            vec![assign(0, Expression::ThreadgroupBarrier)],
            Terminator::Goto(2),
        ),
        block(vec![], Terminator::Return),
    ])
}

/// `shuffle_xor(1, coordinate)` over the SIMD coordinate `builtin`.
fn shuffle_control(builtin: SimdBuiltin) -> Module {
    let shuffle = Expression::Simd {
        op: SimdOp::ShuffleXor,
        arguments: vec![uint(1), Operand::local(2)],
    };
    kernel_of(vec![block(
        vec![
            assign(2, Expression::SimdCoordinate(builtin)),
            assign(5, shuffle),
        ],
        Terminator::Return,
    )])
}

#[test]
fn loop_carried_lane_control_reaches_matrix_operations_before_the_exit_test() {
    validate(&exit_after_matrices(SimdBuiltin::Group)).unwrap();
    let error = validate(&exit_after_matrices(SimdBuiltin::Lane)).unwrap_err();
    assert!(error.message.contains("uniform participation"), "{error}");
}

#[test]
fn group_uniform_control_never_admits_a_barrier() {
    validate(&barrier_under_branch(SimdBuiltin::Count)).unwrap();
    for builtin in [SimdBuiltin::Group, SimdBuiltin::Lane] {
        let error = validate(&barrier_under_branch(builtin)).unwrap_err();
        assert!(error.message.contains("uniform participation"), "{error}");
    }
}

#[test]
fn group_uniform_shuffle_controls_are_rejected() {
    validate(&shuffle_control(SimdBuiltin::Count)).unwrap();
    for builtin in [SimdBuiltin::Group, SimdBuiltin::Lane] {
        let error = validate(&shuffle_control(builtin)).unwrap_err();
        assert!(error.message.contains("uniform participation"), "{error}");
    }
}
