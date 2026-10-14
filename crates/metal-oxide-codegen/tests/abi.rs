#[allow(dead_code)]
mod support;

use metal_oxide_artifact::{Access, Layout, ParameterType, Scalar};
use metal_oxide_ir::{self as ir, Block, Function, Module, SourceLocation, Terminator, Type};

fn module(parameters: Vec<Type>) -> Module {
    let source = SourceLocation {
        file: "kernel.rs".into(),
        line: 1,
        column: 1,
    };
    let count = parameters.len();
    let mut locals = vec![Type::Unit];
    locals.extend(parameters);
    Module {
        types: ir::TypeTable::default(),
        functions: vec![Function {
            name: "vec_add".into(),
            kernel: true,
            required_block: None,
            requires: Vec::new(),
            parameters: count,
            locals,
            blocks: vec![Block {
                statements: vec![],
                terminator: Terminator::Return,
                source: source.clone(),
            }],
            source,
        }],
    }
}

#[test]
fn buffer_and_threadgroup_tables_have_separate_slot_limits() {
    let shared = Type::Buffer {
        element: ir::Element::Scalar(ir::Scalar::U64),
        access: ir::Access::ReadWrite,
        address_space: ir::AddressSpace::Threadgroup,
    };
    // The buffer table keeps its last three slots for the gate, failure and length words.
    let parameters = (0..28)
        .flat_map(|_| [shared, Type::Scalar(ir::Scalar::U32)])
        .chain([shared, shared, shared])
        .collect::<Vec<_>>();
    let valid = module(parameters.clone());
    let codegen = metal_oxide_codegen::Codegen::new(&valid).unwrap();
    let abi = codegen.abi().unwrap();
    assert_eq!(abi.required_features, ["dynamic_threadgroup_memory"]);
    assert_eq!(
        abi.kernels[0]
            .parameters
            .iter()
            .map(|p| p.binding)
            .collect::<Vec<_>>(),
        (0..28)
            .flat_map(|i| [i, i])
            .chain([28, 29, 30])
            .collect::<Vec<_>>()
    );
    assert_eq!(abi.kernels[0].gate_binding, 28);
    assert!(!abi.kernels[0].fallible);
    for (extra, limit) in [(shared, "31"), (Type::Scalar(ir::Scalar::U32), "28")] {
        let mut invalid = parameters.clone();
        invalid.push(extra);
        assert!(
            metal_oxide_codegen::Codegen::new(&module(invalid))
                .unwrap_err()
                .message
                .contains(limit)
        );
    }
    let mut invalid = abi.clone();
    invalid.kernels[0].parameters[2].binding = 0;
    assert!(invalid.validate().is_err());
    let mut invalid = abi.clone();
    invalid.required_features.clear();
    assert!(invalid.validate().is_err());
    let mut invalid = abi;
    let ParameterType::Threadgroup { stride, .. } = &mut invalid.kernels[0].parameters[0].ty else {
        panic!()
    };
    *stride = 4;
    assert!(invalid.validate().is_err());
}

#[test]
fn abi_slots_match_msl_and_typed_host_arguments() {
    let buffer = |access| Type::Buffer {
        element: ir::Element::Scalar(ir::Scalar::F32),
        access,
        address_space: ir::AddressSpace::Device,
    };
    let module = module(vec![
        buffer(ir::Access::Read),
        buffer(ir::Access::Write),
        Type::Scalar(ir::Scalar::U32),
        buffer(ir::Access::ReadWrite),
    ]);
    let codegen = metal_oxide_codegen::Codegen::new(&module).unwrap();
    let abi = codegen.abi().unwrap();
    let msl = codegen.emit().unwrap();
    for parameter in &abi.kernels[0].parameters {
        assert!(msl.contains(&format!("[[buffer({})]]", parameter.binding)));
    }
    assert_eq!(
        abi.kernels[0].parameters[0].ty,
        ParameterType::Buffer {
            element: Layout::scalar(Scalar::F32),
            stride: 4,
            access: Access::Read
        }
    );
    assert_eq!(
        abi.kernels[0].parameters[2].ty,
        ParameterType::Value {
            layout: Layout::scalar(Scalar::U32)
        }
    );
    assert_eq!(
        abi.kernels[0].parameters[3].ty,
        ParameterType::Buffer {
            element: Layout::scalar(Scalar::F32),
            stride: 4,
            access: Access::ReadWrite,
        }
    );
}

#[test]
fn handles_no_parameters_and_rejects_bool_parameters() {
    let empty = metal_oxide_codegen::Codegen::new(&module(vec![]))
        .unwrap()
        .abi()
        .unwrap();
    assert!(empty.kernels[0].parameters.is_empty());
    assert!(
        metal_oxide_codegen::Codegen::new(&module(vec![Type::Scalar(ir::Scalar::Bool)])).is_err()
    );
}

#[test]
fn kernel_interfaces_keep_mixed_bindings_independent_from_helpers() {
    let buffer = |element, access| Type::Buffer {
        element: ir::Element::Scalar(element),
        access,
        address_space: ir::AddressSpace::Device,
    };
    let mut module = module(vec![
        Type::Scalar(ir::Scalar::U8),
        buffer(ir::Scalar::F32, ir::Access::Read),
        Type::Scalar(ir::Scalar::U16),
        buffer(ir::Scalar::U32, ir::Access::Atomic),
        Type::Scalar(ir::Scalar::I32),
        buffer(ir::Scalar::F32, ir::Access::Write),
    ]);
    let mut helper = module.functions[0].clone();
    helper.name = "helper".into();
    helper.kernel = false;
    helper.parameters = 0;
    helper.locals = vec![Type::Unit];
    module.functions.insert(0, helper);
    let mut second = module.functions[1].clone();
    second.name = "scale".into();
    second.parameters = 2;
    second.locals = vec![
        Type::Unit,
        Type::Scalar(ir::Scalar::F32),
        buffer(ir::Scalar::I32, ir::Access::Atomic),
    ];
    second.required_block = Some([32, 1, 1]);
    module.functions.push(second);

    let codegen = metal_oxide_codegen::Codegen::new(&module).unwrap();
    let abi = codegen.abi().unwrap();
    let msl = codegen.emit().unwrap();
    assert_eq!(abi.kernels.len(), 2);
    assert_eq!(abi.kernels[1].required_block, Some([32, 1, 1]));
    for kernel in &abi.kernels {
        let signature = msl
            .lines()
            .find(|line| line.starts_with(&format!("kernel void {}(", kernel.name)))
            .unwrap();
        for (index, parameter) in kernel.parameters.iter().enumerate() {
            assert_eq!(parameter.binding, index as u32);
            assert!(
                signature.contains(&format!(
                    "metal_oxide_arg_{} [[buffer({})]]",
                    index + 1,
                    parameter.binding
                )),
                "{signature}"
            );
        }
    }
    assert!(msl.contains("constant uchar & metal_oxide_arg_1"));
    assert!(msl.contains("constant ushort & metal_oxide_arg_3"));
    assert!(msl.contains("device atomic_uint * metal_oxide_arg_4"));
    assert!(msl.contains("device atomic_int * metal_oxide_arg_2"));
    assert_eq!(codegen.abi().unwrap(), abi);
    abi.validate().unwrap();
}

#[test]
fn kernel_binding_limit_is_checked_before_emission() {
    let mut valid = module(vec![Type::Scalar(ir::Scalar::U32); 28]);
    let codegen = metal_oxide_codegen::Codegen::new(&valid).unwrap();
    let msl = codegen.emit().unwrap();
    assert!(msl.contains("metal_oxide_arg_28 [[buffer(27)]]"), "{msl}");
    assert!(msl.contains("metal_oxide_gate [[buffer(28)]]"), "{msl}");
    assert!(!msl.contains("metal_oxide_failure ["), "{msl}");
    codegen.abi().unwrap().validate().unwrap();
    let function = &mut valid.functions[0];
    let source = function.source.clone();
    function.locals.push(Type::Scalar(ir::Scalar::Bool));
    function.blocks.insert(
        0,
        Block {
            statements: vec![ir::Statement {
                destination: ir::Place::local(29),
                value: ir::Expression::Binary(
                    ir::BinaryOp::Lt,
                    ir::Operand::local(28),
                    ir::Operand::Constant(ir::Constant::U32(5)),
                ),
                source: source.clone(),
            }],
            terminator: Terminator::Assert {
                condition: ir::Operand::local(29),
                expected: true,
                enabled: true,
                target: 1,
                check: ir::Check::Bounds,
            },
            source,
        },
    );
    let codegen = metal_oxide_codegen::Codegen::new(&valid).unwrap();
    let msl = codegen.emit().unwrap();
    assert!(msl.contains("metal_oxide_failure [[buffer(29)]]"), "{msl}");
    let abi = codegen.abi().unwrap();
    assert!(abi.kernels[0].fallible);
    abi.validate().unwrap();
    let invalid = module(vec![Type::Scalar(ir::Scalar::U32); 29]);
    let error = metal_oxide_codegen::Codegen::new(&invalid).unwrap_err();
    assert!(error.message.contains("28"));
}

#[test]
fn kernel_names_cannot_redefine_msl_scalar_aliases() {
    for name in ["uchar", "ushort", "uint", "ulong", "size_t", "ptrdiff_t"] {
        let mut module = module(vec![]);
        module.functions[0].name = name.into();
        let error = metal_oxide_codegen::Codegen::new(&module).unwrap_err();
        assert!(error.message.contains("MSL identifier"), "{name}: {error}");
    }
}

#[test]
fn nested_record_layout_matches_the_emitted_msl_declarations() {
    use ir::{Aggregate, RecordField};
    let mut module = module(vec![]);
    let position = module
        .types
        .intern(Aggregate::Tuple(vec![Type::Scalar(ir::Scalar::F32); 3]));
    let velocity = module.types.intern(Aggregate::Array {
        element: Type::Scalar(ir::Scalar::F32),
        length: 2,
    });
    let particle = module.types.intern(Aggregate::Record {
        name: "Particle".into(),
        fields: [
            ("tag", Type::Scalar(ir::Scalar::U8)),
            ("flags", Type::Scalar(ir::Scalar::U16)),
            ("id", Type::Scalar(ir::Scalar::U32)),
            ("position", position),
            ("velocity", velocity),
        ]
        .into_iter()
        .map(|(name, ty)| RecordField {
            name: name.into(),
            ty,
        })
        .collect(),
    });
    module.functions[0].parameters = 1;
    module.functions[0].locals.push(particle);
    let codegen = metal_oxide_codegen::Codegen::new(&module).unwrap();
    let abi = codegen.abi().unwrap();
    let ParameterType::Value { layout } = &abi.kernels[0].parameters[0].ty else {
        panic!()
    };
    assert_eq!((layout.size, layout.alignment), (28, 4));
    let Type::Aggregate(id) = particle else {
        panic!()
    };
    let main = format!(
        "metal_oxide_aggregate_{id} value = {{}}; std::cout << sizeof(value) << ',' << alignof(decltype(value)); for (const void* field : {{(void*)&value.f0, (void*)&value.f1, (void*)&value.f2, (void*)&value.f3, (void*)&value.f4}}) std::cout << ',' << ((const char*)field - (const char*)&value);"
    );
    assert_eq!(support::execute(&module, &main), "28,4,0,2,4,8,20");
}

#[test]
fn a_kernel_that_reads_lengths_binds_them_after_its_failure_word() {
    let buffer = Type::Buffer {
        element: ir::Element::Scalar(ir::Scalar::F32),
        access: ir::Access::Read,
        address_space: ir::AddressSpace::Device,
    };
    let mut module = module(vec![buffer, Type::Scalar(ir::Scalar::U32)]);
    let function = &mut module.functions[0];
    let source = function.source.clone();
    let usize = Type::Scalar(ir::Scalar::Usize);
    function
        .locals
        .extend([usize, Type::Scalar(ir::Scalar::Bool)]);
    let statement = |destination, value| ir::Statement {
        destination: ir::Place::local(destination),
        value,
        source: source.clone(),
    };
    function.blocks = vec![
        Block {
            statements: vec![
                statement(3, ir::Expression::Len(ir::Operand::local(1))),
                statement(
                    4,
                    ir::Expression::Binary(
                        ir::BinaryOp::Lt,
                        ir::Operand::local(3),
                        ir::Operand::Constant(ir::Constant::Usize(9)),
                    ),
                ),
            ],
            terminator: Terminator::Assert {
                condition: ir::Operand::local(4),
                expected: true,
                enabled: true,
                target: 1,
                check: ir::Check::Bounds,
            },
            source: source.clone(),
        },
        Block {
            statements: vec![],
            terminator: Terminator::Return,
            source,
        },
    ];
    let codegen = metal_oxide_codegen::Codegen::new(&module).unwrap();
    let msl = codegen.emit().unwrap();
    assert!(msl.contains("metal_oxide_gate [[buffer(2)]]"), "{msl}");
    assert!(msl.contains("metal_oxide_failure [[buffer(3)]]"), "{msl}");
    assert!(
        msl.contains("constant ulong* metal_oxide_lengths [[buffer(4)]]"),
        "{msl}"
    );
    assert!(msl.contains("v3 = metal_oxide_lengths[0];"), "{msl}");
    let abi = codegen.abi().unwrap();
    assert!(abi.kernels[0].fallible && abi.kernels[0].lengths);
    abi.validate().unwrap();
}
