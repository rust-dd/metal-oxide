use metal_oxide_ir::*;

fn source() -> SourceLocation {
    SourceLocation {
        file: "places.rs".into(),
        line: 1,
        column: 1,
    }
}

fn place(local: usize, projection: &[u32]) -> Place {
    Place {
        local,
        projection: projection.to_vec(),
    }
}

fn module(complete: bool) -> Module {
    let mut types = TypeTable::default();
    let tuple = types.intern(Aggregate::Tuple(vec![Type::Scalar(Scalar::U32); 2]));
    let record = types.intern(Aggregate::Record(vec![tuple, Type::Scalar(Scalar::U32)]));
    let mut statements = vec![
        Statement {
            destination: place(1, &[0, 0]),
            value: Expression::Use(Operand::Constant(Constant::U32(2))),
            source: source(),
        },
        Statement {
            destination: place(1, &[1]),
            value: Expression::Use(Operand::Constant(Constant::U32(7))),
            source: source(),
        },
    ];
    if complete {
        statements.push(Statement {
            destination: place(1, &[0, 1]),
            value: Expression::Use(Operand::Constant(Constant::U32(5))),
            source: source(),
        });
    }
    statements.push(Statement {
        destination: Place::local(0),
        value: Expression::Use(Operand::Place(place(1, &[]))),
        source: source(),
    });
    Module {
        types,
        functions: vec![Function {
            name: "partial".into(),
            kernel: false,
            required_block: None,
            parameters: 0,
            locals: vec![record, record],
            blocks: vec![Block {
                statements,
                terminator: Terminator::Return,
                source: source(),
            }],
            source: source(),
        }],
    }
}

#[test]
fn nested_field_writes_initialize_a_complete_owned_value() {
    validate(&module(true)).unwrap();
}

#[test]
fn whole_value_read_rejects_an_uninitialized_nested_field() {
    let error = validate(&module(false)).unwrap_err();
    assert!(error.message.contains("uninitialized"), "{error}");
}

#[test]
fn projected_read_does_not_require_unrelated_fields() {
    let mut module = module(false);
    let f = &mut module.functions[0];
    f.locals[0] = Type::Scalar(Scalar::U32);
    f.blocks[0].statements.last_mut().unwrap().value =
        Expression::Use(Operand::Place(place(1, &[0, 0])));
    validate(&module).unwrap();
}

#[test]
fn initialization_at_a_join_requires_each_nested_field_on_both_paths() {
    for complete in [true, false] {
        let mut module = module(true);
        let f = &mut module.functions[0];
        let statements = std::mem::take(&mut f.blocks[0].statements);
        let assignments = statements[..3].to_vec();
        let mut other = assignments.clone();
        if !complete {
            other.pop();
        }
        f.blocks = vec![
            Block {
                statements: vec![],
                terminator: Terminator::Branch {
                    condition: Operand::Constant(Constant::Bool(true)),
                    then_block: 1,
                    else_block: 2,
                },
                source: source(),
            },
            Block {
                statements: assignments,
                terminator: Terminator::Goto(3),
                source: source(),
            },
            Block {
                statements: other,
                terminator: Terminator::Goto(3),
                source: source(),
            },
            Block {
                statements: vec![statements[3].clone()],
                terminator: Terminator::Return,
                source: source(),
            },
        ];
        assert_eq!(validate(&module).is_ok(), complete);
    }
}
