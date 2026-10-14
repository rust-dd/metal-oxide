use super::Mirrors;
use metal_oxide_artifact::{
    Abi, Access, FieldLayout, Kernel, Layout, LayoutKind, MathMode, Parameter, ParameterType,
    Scalar,
};

fn record(name: &str, fields: Vec<(&str, u64, Layout)>) -> Layout {
    let size = fields
        .iter()
        .map(|(_, offset, field)| offset + field.size)
        .max()
        .unwrap();
    Layout {
        size,
        alignment: 4,
        kind: LayoutKind::Record {
            name: name.into(),
            fields: fields
                .into_iter()
                .map(|(name, offset, layout)| FieldLayout {
                    name: name.into(),
                    offset,
                    layout,
                })
                .collect(),
        },
    }
}

fn abi(kernel: &str, parameters: Vec<(&str, ParameterType)>) -> Abi {
    Abi {
        version: metal_oxide_artifact::ABI_VERSION,
        math_mode: MathMode::Precise,
        required_features: Vec::new(),
        kernels: vec![Kernel {
            name: kernel.into(),
            parameters: parameters
                .into_iter()
                .enumerate()
                .map(|(binding, (name, ty))| Parameter {
                    name: name.into(),
                    binding: binding as u32,
                    ty,
                })
                .collect(),
            required_block: None,
            gate_binding: 2,
            fallible: false,
            lengths: false,
            one_dimensional: false,
            safe: false,
            requires: Vec::new(),
        }],
    }
}

/// Generated tokens without the spaces that their printing inserts.
fn text(tokens: proc_macro2::TokenStream) -> String {
    tokens.to_string().replace(' ', "")
}

#[test]
fn records_mirror_their_layout_and_avoid_generic_names() {
    let float = Layout::scalar(Scalar::F32);
    let pair = Layout {
        size: 8,
        alignment: 4,
        kind: LayoutKind::Array {
            element: Box::new(float.clone()),
            length: 2,
            stride: 4,
        },
    };
    let particle = record(
        "MetalOxideBuffer0",
        vec![("x", 0, float.clone()), ("v", 4, pair)],
    );
    let abi = abi(
        "type",
        vec![
            (
                "particles",
                ParameterType::Buffer {
                    element: particle.clone(),
                    stride: 12,
                    access: Access::Read,
                },
            ),
            ("config", ParameterType::Value { layout: particle }),
        ],
    );
    let mirrors = Mirrors::new(&abi).unwrap();
    let records = text(mirrors.definitions().unwrap());
    assert!(records.contains("pubstructMetalOxideBuffer0Record{pubx:::core::primitive::f32,pubv:[::core::primitive::f32;2usize]}"));
    assert!(records.contains("Layout::record(\"MetalOxideBuffer0\""));
    assert!(records.contains("offset_of!(MetalOxideBuffer0Record,v)==4usize"));
    let launcher = text(
        mirrors
            .kernel(&abi.kernels[0])
            .unwrap()
            .launcher(&quote::format_ident!("LIBRARY")),
    );
    assert!(
        launcher.contains("pubunsafefnr#type<MetalOxideBuffer0>"),
        "{launcher}"
    );
    assert!(
        launcher.contains("MetalOxideBuffer0:::metal_oxide::ReadArgument<MetalOxideBuffer0Record>")
    );
    assert!(launcher.contains("argument_1:MetalOxideBuffer0Record"));
    assert!(launcher.contains("::metal_oxide::Argument::value(*argument_1)?"));
}
