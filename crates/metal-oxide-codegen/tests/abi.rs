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
        records: vec![],
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
    let abi = metal_oxide_codegen::abi(&module).unwrap();
    let msl = metal_oxide_codegen::emit(&module).unwrap();
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
    let mut abi = metal_oxide_codegen::abi(&module(vec![])).unwrap();
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
    let empty = metal_oxide_codegen::abi(&module(vec![])).unwrap();
    assert!(empty.kernels[0].parameters.is_empty());
    metal_oxide_codegen::bindings(&empty).unwrap();
    let mut abi = metal_oxide_codegen::abi(&module(vec![Type::Scalar(ir::Scalar::U32)])).unwrap();
    abi.kernels[0].name = "type".into();
    abi.kernels[0].parameters[0].name = "config".into();
    let source = metal_oxide_codegen::bindings(&abi).unwrap();
    assert!(source.contains("fn r#type"));
    assert!(source.contains("r#arg_0_config: u32"));
    assert!(metal_oxide_codegen::abi(&module(vec![Type::Scalar(ir::Scalar::Bool)])).is_err());
}
