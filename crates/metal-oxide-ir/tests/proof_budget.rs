use metal_oxide_ir::*;

const KERNELS: usize = 600;

fn source() -> SourceLocation {
    SourceLocation {
        file: "proof_budget.rs".into(),
        line: 1,
        column: 1,
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

fn constant(value: u32) -> Operand {
    Operand::Constant(Constant::U32(value))
}

fn bounded_loop(name: String, kernel: bool) -> Function {
    Function {
        name,
        kernel,
        required_block: None,
        parameters: 0,
        locals: vec![
            Type::Unit,
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::Bool),
        ],
        blocks: vec![
            block(
                vec![assign(1, Expression::Use(constant(0)))],
                Terminator::Goto(1),
            ),
            block(
                vec![assign(
                    2,
                    Expression::Binary(BinaryOp::Lt, Operand::local(1), constant(64)),
                )],
                Terminator::Branch {
                    condition: Operand::local(2),
                    then_block: 2,
                    else_block: 4,
                },
            ),
            block(
                vec![],
                Terminator::Assert {
                    condition: Operand::local(2),
                    expected: true,
                    enabled: true,
                    target: 3,
                    check: Check::Overflow(OverflowOp::Add),
                },
            ),
            block(
                vec![assign(
                    1,
                    Expression::Binary(BinaryOp::Add, Operand::local(1), constant(1)),
                )],
                Terminator::Goto(1),
            ),
            block(vec![], Terminator::Return),
        ],
        source: source(),
    }
}

fn caller(name: &str, functions: impl IntoIterator<Item = (usize, Vec<Operand>)>) -> Function {
    Function {
        name: name.into(),
        kernel: true,
        required_block: None,
        parameters: 0,
        locals: vec![Type::Unit],
        blocks: vec![block(
            functions
                .into_iter()
                .map(|(function, arguments)| {
                    assign(
                        0,
                        Expression::Call {
                            function,
                            arguments,
                        },
                    )
                })
                .collect(),
            Terminator::Return,
        )],
        source: source(),
    }
}

#[test]
fn independent_kernels_do_not_spend_each_others_proof_budget() {
    // Each bounded loop needs fewer than 300 block visits. Together these
    // exports exceed 100,000 visits, without any entrypoint approaching it.
    let module = Module {
        types: TypeTable::default(),
        functions: (0..KERNELS)
            .map(|index| bounded_loop(format!("kernel_{index}"), true))
            .collect(),
    };
    validate(&module).unwrap();
}

#[test]
fn one_kernel_cannot_reset_its_budget_by_calling_helpers() {
    let mut functions = (0..KERNELS)
        .map(|index| bounded_loop(format!("helper_{index}"), false))
        .collect::<Vec<_>>();
    functions.push(caller("kernel", (0..KERNELS).map(|index| (index, vec![]))));
    let error = validate(&Module {
        types: TypeTable::default(),
        functions,
    })
    .unwrap_err();
    assert!(
        error.message.contains("proof analysis budget exceeded"),
        "{error}"
    );
}

#[test]
fn cached_helper_proofs_keep_their_argument_facts() {
    let helper = Function {
        name: "checked_helper".into(),
        kernel: false,
        required_block: None,
        parameters: 1,
        locals: vec![
            Type::Unit,
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::Bool),
        ],
        blocks: vec![
            block(
                vec![assign(
                    2,
                    Expression::Binary(BinaryOp::Lt, Operand::local(1), constant(4)),
                )],
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
    for invalid in [false, true] {
        let module = Module {
            types: TypeTable::default(),
            functions: vec![
                helper.clone(),
                caller("first", [(0, vec![constant(3)])]),
                caller("second", [(0, vec![constant(if invalid { 4 } else { 3 })])]),
            ],
        };
        // The helper is emitted once, so a context that cannot prove its check makes it fallible.
        let analyses = validate(&module).unwrap();
        assert_eq!(analyses[0].checks.contains(&0), invalid);
        for function in &analyses {
            assert_eq!(function.fallible, invalid);
        }
    }
}
