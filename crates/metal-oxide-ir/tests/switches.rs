use metal_oxide_ir::*;

fn source() -> SourceLocation {
    SourceLocation {
        file: "match.rs".into(),
        line: 4,
        column: 5,
    }
}

fn module() -> Module {
    let block = |terminator, value: Option<u32>| Block {
        statements: value
            .map(|value| Statement {
                destination: Place::local(0),
                value: Expression::Use(Operand::Constant(Constant::U32(value))),
                source: source(),
            })
            .into_iter()
            .collect(),
        terminator,
        source: source(),
    };
    Module {
        types: TypeTable::default(),
        functions: vec![Function {
            name: "classify".into(),
            kernel: false,
            required_block: None,
            parameters: 1,
            locals: vec![Type::Scalar(Scalar::U32); 3],
            source: source(),
            blocks: vec![
                block(
                    Terminator::Switch {
                        discriminant: Operand::local(1),
                        cases: vec![
                            (Constant::U32(0), 1),
                            (Constant::U32(1), 1),
                            (Constant::U32(7), 2),
                        ],
                        otherwise: 3,
                    },
                    None,
                ),
                block(Terminator::Return, Some(11)),
                block(Terminator::Return, Some(22)),
                block(Terminator::Return, Some(33)),
            ],
        }],
    }
}

#[test]
fn integer_switch_has_unique_successors_and_valid_types() {
    let module = module();
    validate(&module).unwrap();
    assert_eq!(
        module.functions[0].blocks[0].terminator.successors(),
        [1, 2, 3]
    );
}

#[test]
fn case_type_must_match_the_discriminant() {
    let mut module = module();
    if let Terminator::Switch { cases, .. } = &mut module.functions[0].blocks[0].terminator {
        cases[0].0 = Constant::I32(0);
    }
    let error = validate(&module).unwrap_err();
    assert_eq!(error.source, source());
    assert!(error.message.contains("case type"));
}

#[test]
fn duplicate_case_values_are_rejected() {
    let mut module = module();
    if let Terminator::Switch { cases, .. } = &mut module.functions[0].blocks[0].terminator {
        cases.push((Constant::U32(0), 2));
    }
    assert!(validate(&module).unwrap_err().message.contains("duplicate"));
}

#[test]
fn noninteger_discriminants_are_rejected() {
    let mut module = module();
    module.functions[0].locals[1] = Type::Scalar(Scalar::F32);
    assert!(validate(&module).unwrap_err().message.contains("integer"));
}

#[test]
fn uninitialized_switch_input_is_rejected() {
    let mut module = module();
    if let Terminator::Switch { discriminant, .. } = &mut module.functions[0].blocks[0].terminator {
        *discriminant = Operand::local(2);
    }
    assert!(
        validate(&module)
            .unwrap_err()
            .message
            .contains("uninitialized")
    );
}

#[test]
fn invalid_default_and_case_targets_are_rejected() {
    for default in [false, true] {
        let mut module = module();
        if let Terminator::Switch {
            cases, otherwise, ..
        } = &mut module.functions[0].blocks[0].terminator
        {
            if default {
                *otherwise = 99;
            } else {
                cases[0].1 = 99;
            }
        }
        assert!(
            validate(&module)
                .unwrap_err()
                .message
                .contains("block target")
        );
    }
}

#[test]
fn every_case_must_initialize_the_return_place() {
    let mut module = module();
    module.functions[0].blocks[2].statements.clear();
    assert!(
        validate(&module)
            .unwrap_err()
            .message
            .contains("uninitialized return")
    );
}
