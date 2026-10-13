use crate::host_types::HostTypes;
use metal_oxide_artifact::{Abi, Access, Error, Kernel, ParameterType};
use std::{collections::HashSet, fmt::Write};

pub fn bindings(abi: &Abi) -> Result<String, Error> {
    abi.validate()?;
    let types = HostTypes::new(abi)?;
    let mut output = String::new();
    types.emit(&mut output);
    output.push_str("pub struct Kernels {\n    device: metal_oxide::Device,\n    module: metal_oxide::Module,\n");
    for index in 0..abi.kernels.len() {
        writeln!(
            output,
            "    pipeline_{index}: std::sync::OnceLock<metal_oxide::Pipeline>,"
        )
        .unwrap();
    }
    output.push_str("}\n\n#[allow(dead_code)]\npub fn load(device: &metal_oxide::Device, directory: impl std::convert::AsRef<std::path::Path>) -> metal_oxide::Result<Kernels> {\n    metal_oxide_kernels(device, metal_oxide::Module::from_artifact(device, directory)?)\n}\n");
    output.push_str("\n#[allow(dead_code)]\npub fn load_bytes(device: &metal_oxide::Device, manifest: &str, metallib: &[u8]) -> metal_oxide::Result<Kernels> {\n    metal_oxide_kernels(device, metal_oxide::Module::from_bytes(device, manifest, metallib)?)\n}\n");
    output.push_str("\nfn metal_oxide_kernels(device: &metal_oxide::Device, module: metal_oxide::Module) -> metal_oxide::Result<Kernels> {\n");
    writeln!(
        output,
        "    module.verify_abi({:?})?;",
        metal_oxide_artifact::sha256(abi.to_json()?.as_bytes())
    )
    .unwrap();
    output.push_str("    Ok(Kernels {\n        device: device.clone(),\n        module,\n");
    for index in 0..abi.kernels.len() {
        writeln!(
            output,
            "        pipeline_{index}: std::sync::OnceLock::new(),"
        )
        .unwrap();
    }
    output.push_str("    })\n}\n");
    output.push_str(
        r#"
fn metal_oxide_pipeline<'a>(
    device: &metal_oxide::Device,
    module: &metal_oxide::Module,
    slot: &'a std::sync::OnceLock<metal_oxide::Pipeline>,
    name: &str,
) -> metal_oxide::Result<&'a metal_oxide::Pipeline> {
    if let Some(pipeline) = slot.get() {
        return Ok(pipeline);
    }
    let pipeline = metal_oxide::Pipeline::new(device, module, name)?;
    Ok(slot.get_or_init(|| pipeline))
}

#[allow(dead_code)]
impl Kernels {
"#,
    );
    for (index, kernel) in abi.kernels.iter().enumerate() {
        emit_method(&mut output, &types, index, kernel)?;
    }
    output.push_str("}\n");
    Ok(output)
}

/// The generic type of a buffer parameter; host types reserve these names.
pub(crate) fn buffer_type(binding: u32) -> String {
    format!("MetalOxideBuffer{binding}")
}

/// One method per kernel that returns its launch operation; buffers are generic over the
/// borrowed, owned and shared argument forms.
fn emit_method(
    output: &mut String,
    types: &HostTypes,
    index: usize,
    kernel: &Kernel,
) -> Result<(), Error> {
    if ["self", "Self", "crate", "super", "_"].contains(&kernel.name.as_str()) {
        return Err(Error(
            "kernel name cannot be represented in Rust bindings".into(),
        ));
    }
    let mut names =
        HashSet::from(["self", "Self", "crate", "super", "config", "_"].map(str::to_owned));
    let mut parameters = Vec::new();
    let mut generics = Vec::new();
    let mut arguments = Vec::new();
    for parameter in &kernel.parameters {
        let mut name = parameter.name.clone();
        while !names.insert(name.clone()) {
            name = format!("arg_{}_{name}", parameter.binding);
        }
        let (ty, argument) = match &parameter.ty {
            ParameterType::Value { layout } => (
                types.name(layout),
                format!("metal_oxide::Argument::value(*r#{name})?"),
            ),
            ParameterType::Threadgroup { element, .. } => (
                "usize".into(),
                format!(
                    "metal_oxide::Argument::threadgroup::<{}>(*r#{name})?",
                    types.name(element)
                ),
            ),
            ParameterType::Buffer {
                element, access, ..
            } => {
                let generic = buffer_type(parameter.binding);
                let (form, view, method) = match access {
                    Access::Read => ("ReadArgument", "buffer", "read"),
                    Access::Write => ("WriteArgument", "buffer_mut", "write"),
                    Access::ReadWrite => ("WriteArgument", "buffer_mut", "read_write"),
                    Access::Atomic => ("WriteArgument", "buffer_mut", "atomic"),
                };
                generics.push(format!(
                    "{generic}: metal_oxide::{form}<{}>",
                    types.name(element)
                ));
                (
                    generic,
                    format!(
                        "metal_oxide::Argument::{method}(metal_oxide::{form}::{view}(r#{name}))"
                    ),
                )
            }
        };
        parameters.push((name, ty));
        arguments.push(argument);
    }
    let config = match kernel.required_block {
        Some([x, y, z]) => format!("metal_oxide::LaunchConfig<{x}, {y}, {z}>"),
        None => "impl std::convert::Into<metal_oxide::DynamicLaunchConfig>".into(),
    };
    let tuple = |items: Vec<String>| match items.len() {
        1 => format!("({},)", items[0]),
        _ => format!("({})", items.join(", ")),
    };
    let values = tuple(
        parameters
            .iter()
            .map(|(name, _)| format!("r#{name}"))
            .collect(),
    );
    let output_type = tuple(parameters.iter().map(|(_, ty)| ty.clone()).collect());
    output.push_str("    /// # Safety\n    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.\n    #[allow(clippy::too_many_arguments, clippy::type_complexity)]\n");
    write!(output, "    pub unsafe fn r#{}", kernel.name).unwrap();
    if !generics.is_empty() {
        let names = kernel
            .parameters
            .iter()
            .filter(|parameter| matches!(parameter.ty, ParameterType::Buffer { .. }))
            .map(|parameter| buffer_type(parameter.binding))
            .collect::<Vec<_>>();
        write!(output, "<{}>", names.join(", ")).unwrap();
    }
    write!(output, "(&self, config: {config}").unwrap();
    for (name, ty) in &parameters {
        write!(output, ", r#{name}: {ty}").unwrap();
    }
    write!(
        output,
        ") -> metal_oxide::Launch<{output_type}, {}>",
        arguments.len()
    )
    .unwrap();
    if !generics.is_empty() {
        write!(
            output,
            "\n    where\n        {},",
            generics.join(",\n        ")
        )
        .unwrap();
    }
    output.push_str("\n    {\n        // SAFETY: kernel-specific preconditions are delegated to this function's caller.\n        unsafe {\n");
    writeln!(
        output,
        "            metal_oxide::Launch::new(\n                self::metal_oxide_pipeline(&self.device, &self.module, &self.pipeline_{index}, {:?}),\n                config,\n                {values},\n                |{values}| Ok([{}]),\n            )",
        kernel.name,
        arguments.join(", ")
    )
    .unwrap();
    output.push_str("        }\n    }\n\n");
    Ok(())
}
