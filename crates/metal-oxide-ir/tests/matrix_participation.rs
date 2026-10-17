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
    let below = Expression::Binary(BinaryOp::Lt, Operand::local(1), uint(1));
    let assertion = Terminator::Assert {
        condition: Operand::local(2),
        expected: true,
        enabled: true,
        target: 1,
        check: Check::Overflow(OverflowOp::Add),
    };
    let output = Type::Buffer {
        element: Element::Scalar(Scalar::F32),
        access: Access::Write,
        address_space: AddressSpace::Device,
    };
    let f = vec![
        block(vec![assign(2, below)], assertion),
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
                vec![Type::Unit, output, U32, Type::Matrix(Scalar::F32)],
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
