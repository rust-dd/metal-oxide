use metal_oxide_artifact::{Access, ParameterType, Scalar};
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
fn abi_slots_match_msl_and_typed_host_arguments() {
    let buffer = |access| Type::Buffer {
        element: ir::Scalar::F32,
        access,
        address_space: ir::AddressSpace::Device,
    };
    let module = module(vec![
        buffer(ir::Access::Read),
        buffer(ir::Access::Write),
        Type::Scalar(ir::Scalar::U32),
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
            element: Scalar::F32,
            access: Access::Read
        }
    );
    assert_eq!(
        abi.kernels[0].parameters[2].ty,
        ParameterType::Scalar {
            scalar: Scalar::U32
        }
    );
    let bindings = metal_oxide_codegen::bindings(&abi).unwrap();
    assert!(bindings.contains("&metal_oxide::Buffer<f32>"));
    assert!(bindings.contains("&mut metal_oxide::Buffer<f32>"));
    assert!(bindings.contains("r#arg_2: u32"));
    assert!(bindings.contains("pub unsafe fn r#vec_add"));
    assert!(
        bindings
            .contains("pub unsafe fn r#enqueue_vec_add(&self, batch: &mut metal_oxide::Batch<'_>")
    );
    assert!(bindings.contains("batch.launch(&self.pipeline_0"));
}

#[test]
fn generated_enqueue_names_cannot_shadow_another_entrypoint() {
    let mut abi = metal_oxide_codegen::Codegen::new(&module(vec![]))
        .unwrap()
        .abi()
        .unwrap();
    let mut other = abi.kernels[0].clone();
    other.name = "enqueue_vec_add".into();
    abi.kernels.push(other);
    assert!(
        metal_oxide_codegen::bindings(&abi)
            .unwrap_err()
            .to_string()
            .contains("collide")
    );
}

#[test]
fn handles_no_parameters_and_rust_name_collisions() {
    let empty = metal_oxide_codegen::Codegen::new(&module(vec![]))
        .unwrap()
        .abi()
        .unwrap();
    assert!(empty.kernels[0].parameters.is_empty());
    metal_oxide_codegen::bindings(&empty).unwrap();
    let mut abi = metal_oxide_codegen::Codegen::new(&module(vec![Type::Scalar(ir::Scalar::U32)]))
        .unwrap()
        .abi()
        .unwrap();
    abi.kernels[0].name = "type".into();
    abi.kernels[0].parameters[0].name = "config".into();
    let source = metal_oxide_codegen::bindings(&abi).unwrap();
    assert!(source.contains("fn r#type"));
    assert!(source.contains("r#arg_0_config: u32"));
    assert!(
        metal_oxide_codegen::Codegen::new(&module(vec![Type::Scalar(ir::Scalar::Bool)])).is_err()
    );
}

#[test]
fn kernel_interfaces_keep_mixed_bindings_independent_from_helpers() {
    let buffer = |element, access| Type::Buffer {
        element,
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
    let valid = module(vec![Type::Scalar(ir::Scalar::U32); 31]);
    let codegen = metal_oxide_codegen::Codegen::new(&valid).unwrap();
    assert!(codegen.emit().unwrap().contains("[[buffer(30)]]"));
    codegen.abi().unwrap().validate().unwrap();
    let invalid = module(vec![Type::Scalar(ir::Scalar::U32); 32]);
    let error = metal_oxide_codegen::Codegen::new(&invalid).unwrap_err();
    assert!(error.message.contains("31"));
}
