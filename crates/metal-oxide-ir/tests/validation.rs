use metal_oxide_ir::*;

fn source() -> SourceLocation {
    SourceLocation {
        file: "kernel.rs".into(),
        line: 7,
        column: 5,
    }
}

fn module() -> Module {
    Module {
        types: TypeTable::default(),
        functions: vec![Function {
            name: "sum".into(),
            kernel: false,
            required_block: None,
            parameters: 2,
            source: source(),
            locals: vec![Type::Scalar(Scalar::F32); 3],
            blocks: vec![Block {
                statements: vec![Statement {
                    destination: Place::local(0),
                    value: Expression::Binary(BinaryOp::Add, Operand::local(1), Operand::local(2)),
                    source: source(),
                }],
                terminator: Terminator::Return,
                source: source(),
            }],
        }],
    }
}

#[test]
fn scalar_helper_has_valid_types_and_initialized_return() {
    validate(&module()).unwrap();
}

#[test]
fn invalid_operand_reports_its_source_location() {
    let mut module = module();
    module.functions[0].blocks[0].statements[0].value = Expression::Use(Operand::local(9));
    let error = validate(&module).unwrap_err();
    assert_eq!(error.source, source());
    assert!(error.message.contains("local"));
}

#[test]
fn assignment_cannot_change_a_local_type() {
    let mut module = module();
    module.functions[0].blocks[0].statements[0].value =
        Expression::Use(Operand::Constant(Constant::U32(1)));
    assert!(validate(&module).unwrap_err().message.contains("type"));
}

#[test]
fn return_value_must_be_initialized_on_both_branches() {
    let mut module = module();
    let f = &mut module.functions[0];
    let assignment = f.blocks[0].statements.clone();
    f.locals.push(Type::Scalar(Scalar::Bool));
    f.parameters = 3;
    f.blocks = vec![
        Block {
            statements: vec![],
            terminator: Terminator::Branch {
                condition: Operand::local(3),
                then_block: 1,
                else_block: 2,
            },
            source: source(),
        },
        Block {
            statements: assignment,
            terminator: Terminator::Goto(3),
            source: source(),
        },
        Block {
            statements: vec![],
            terminator: Terminator::Goto(3),
            source: source(),
        },
        Block {
            statements: vec![],
            terminator: Terminator::Return,
            source: source(),
        },
    ];
    assert!(
        validate(&module)
            .unwrap_err()
            .message
            .contains("uninitialized")
    );
}

#[test]
fn write_requires_a_write_buffer() {
    let mut module = module();
    let f = &mut module.functions[0];
    f.locals[0] = Type::Unit;
    f.locals[1] = Type::Buffer {
        element: Scalar::F32,
        access: Access::Read,
        address_space: AddressSpace::Device,
    };
    f.blocks[0].statements = vec![Statement {
        destination: Place::local(0),
        value: Expression::BufferStore {
            buffer: Operand::local(1),
            index: Operand::Constant(Constant::U32(0)),
            value: Operand::local(2),
        },
        source: source(),
    }];
    assert!(validate(&module).unwrap_err().message.contains("write"));
}

#[test]
fn invalid_block_target_is_an_error_instead_of_a_panic() {
    let mut module = module();
    module.functions[0].blocks[0].terminator = Terminator::Goto(17);
    assert!(validate(&module).unwrap_err().message.contains("block"));
}

#[test]
fn recursive_call_graph_is_rejected() {
    let mut module = module();
    module.functions[0].blocks[0].statements[0].value = Expression::Call {
        function: 0,
        arguments: vec![Operand::local(1), Operand::local(2)],
    };
    assert!(validate(&module).unwrap_err().message.contains("recursion"));
}
