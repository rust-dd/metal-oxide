use metal_oxide_ir::*;

fn source() -> SourceLocation {
    SourceLocation {
        file: "aggregate.rs".into(),
        line: 9,
        column: 3,
    }
}

fn identity(types: TypeTable, ty: Type) -> Module {
    Module {
        types,
        functions: vec![Function {
            name: "identity".into(),
            kernel: false,
            required_block: None,
            parameters: 1,
            locals: vec![ty, ty],
            blocks: vec![Block {
                statements: vec![Statement {
                    destination: Place::local(0),
                    value: Expression::Use(Operand::local(1)),
                    source: source(),
                }],
                terminator: Terminator::Return,
                source: source(),
            }],
            source: source(),
        }],
    }
}

fn id(ty: Type) -> usize {
    let Type::Aggregate(id) = ty else {
        panic!("expected aggregate type")
    };
    id
}

#[test]
fn nested_owned_shapes_can_be_passed_and_returned() {
    let mut types = TypeTable::default();
    let array = types.intern(Aggregate::Array {
        element: Type::Scalar(Scalar::U16),
        length: 3,
    });
    let tuple = types.intern(Aggregate::Tuple(vec![array, Type::Scalar(Scalar::U32)]));
    let record = types.intern(Aggregate::record(
        "Record",
        vec![tuple, Type::Scalar(Scalar::Bool)],
    ));
    let module = identity(types, record);
    validate(&module).unwrap();
    assert_eq!(
        operand_type(
            &module,
            &module.functions[0],
            &Operand::Place(Place {
                local: 1,
                projection: vec![Projection::Field(0)]
            }),
            &source(),
        )
        .unwrap(),
        tuple
    );
}

#[test]
fn interned_shape_identity_preserves_kind_length_and_components() {
    let mut types = TypeTable::default();
    let fields = vec![Type::Scalar(Scalar::U32), Type::Scalar(Scalar::Bool)];
    let record = types.intern(Aggregate::record("Record", fields.clone()));
    let tuple = types.intern(Aggregate::Tuple(fields.clone()));
    assert_ne!(record, tuple);
    assert_eq!(record, types.intern(Aggregate::record("Record", fields)));
    let array = types.intern(Aggregate::Array {
        element: record,
        length: 2,
    });
    assert_ne!(
        array,
        types.intern(Aggregate::Array {
            element: record,
            length: 3,
        })
    );
    assert_ne!(
        array,
        types.intern(Aggregate::Array {
            element: tuple,
            length: 2,
        })
    );
    validate(&identity(types, array)).unwrap();
}

#[test]
fn array_construction_checks_every_element_and_arity() {
    for (fields, expected) in [
        (vec![Constant::U32(1), Constant::U32(2)], None),
        (vec![Constant::U32(1)], Some("count")),
        (
            vec![Constant::U32(1), Constant::F32(0)],
            Some("type mismatch"),
        ),
    ] {
        let mut types = TypeTable::default();
        let array = types.intern(Aggregate::Array {
            element: Type::Scalar(Scalar::U32),
            length: 2,
        });
        let mut module = identity(types, array);
        module.functions[0].parameters = 0;
        module.functions[0].blocks[0].statements[0].value = Expression::Aggregate {
            ty: id(array),
            fields: fields.into_iter().map(Operand::Constant).collect(),
        };
        match expected {
            None => validate(&module).unwrap(),
            Some(message) => {
                let error = validate(&module).unwrap_err();
                assert_eq!(error.source, source());
                assert!(error.message.contains(message));
            }
        }
    }
}

#[test]
fn array_updates_validate_the_index_and_replacement_type() {
    for (field, replacement, expected) in [
        (1, Constant::U32(9), None),
        (2, Constant::U32(9), Some("field")),
        (0, Constant::U16(9), Some("type mismatch")),
    ] {
        let mut types = TypeTable::default();
        let array = types.intern(Aggregate::Array {
            element: Type::Scalar(Scalar::U32),
            length: 2,
        });
        let mut module = identity(types, array);
        module.functions[0].blocks[0].statements[0].value = Expression::AggregateUpdate {
            aggregate: Operand::local(1),
            field,
            value: Operand::Constant(replacement),
        };
        match expected {
            None => validate(&module).unwrap(),
            Some(message) => assert!(validate(&module).unwrap_err().message.contains(message)),
        }
    }
}

#[test]
fn nested_component_replacement_keeps_the_exact_shape() {
    let mut types = TypeTable::default();
    let array = types.intern(Aggregate::Array {
        element: Type::Scalar(Scalar::U32),
        length: 2,
    });
    let other = types.intern(Aggregate::Array {
        element: Type::Scalar(Scalar::U32),
        length: 3,
    });
    let record = types.intern(Aggregate::record("Record", vec![array]));
    let mut module = identity(types, record);
    let function = &mut module.functions[0];
    function.parameters = 2;
    function.locals.push(other);
    function.blocks[0].statements[0].value = Expression::AggregateUpdate {
        aggregate: Operand::local(1),
        field: 0,
        value: Operand::local(2),
    };
    assert!(
        validate(&module)
            .unwrap_err()
            .message
            .contains("type mismatch")
    );
}

#[test]
fn recursive_forward_and_missing_type_references_are_rejected() {
    for child in [0, 1, usize::MAX] {
        let mut types = TypeTable::default();
        let record = types.intern(Aggregate::record("Record", vec![Type::Aggregate(child)]));
        let error = validate(&identity(types, record)).unwrap_err();
        assert_eq!(error.source, source());
        assert!(error.message.contains("earlier definitions"));
    }
    let mut types = TypeTable::default();
    let first = types.intern(Aggregate::record("Record", vec![Type::Aggregate(1)]));
    types.intern(Aggregate::Tuple(vec![first]));
    assert!(validate(&identity(types, first)).is_err());
    let error = validate(&identity(TypeTable::default(), Type::Aggregate(0))).unwrap_err();
    assert!(error.message.contains("invalid aggregate type"));
}

#[test]
fn non_value_components_are_rejected_in_nested_storage() {
    for component in [
        Type::Unit,
        Type::Never,
        Type::Checked(Scalar::Bool),
        Type::Checked(Scalar::F32),
        Type::Buffer {
            element: Element::Scalar(Scalar::U32),
            access: Access::Read,
            address_space: AddressSpace::Device,
        },
        Type::Buffer {
            element: Element::Scalar(Scalar::U32),
            access: Access::ReadWrite,
            address_space: AddressSpace::Threadgroup,
        },
    ] {
        let mut types = TypeTable::default();
        let record = types.intern(Aggregate::record("Record", vec![component]));
        let error = validate(&identity(types, record)).unwrap_err();
        assert!(error.message.contains("owned value components"));
    }
}

#[test]
fn empty_owned_aggregates_have_an_explicit_diagnostic() {
    for aggregate in [
        Aggregate::record("Record", vec![]),
        Aggregate::Tuple(vec![]),
        Aggregate::Array {
            element: Type::Scalar(Scalar::U32),
            length: 0,
        },
    ] {
        let mut types = TypeTable::default();
        let ty = types.intern(aggregate);
        let error = validate(&identity(types, ty)).unwrap_err();
        assert!(error.message.contains("empty aggregate"));
    }
}

#[test]
fn aggregate_kernel_parameters_require_canonical_owned_leaves() {
    for scalar in [Scalar::U32, Scalar::Bool] {
        let mut types = TypeTable::default();
        let record = types.intern(Aggregate::record("Record", vec![Type::Scalar(scalar)]));
        let mut module = identity(types, record);
        let function = &mut module.functions[0];
        function.kernel = true;
        function.locals[0] = Type::Unit;
        function.blocks[0].statements.clear();
        assert_eq!(validate(&module).is_ok(), scalar == Scalar::U32);
    }
}
