use crate::host_types::HostTypes;
use metal_oxide_artifact::{Abi, Access, Error, Kernel, ParameterType};
use std::{collections::HashSet, fmt::Write};

pub fn bindings(abi: &Abi) -> Result<String, Error> {
    abi.validate()?;
    let mut methods = HashSet::new();
    for kernel in &abi.kernels {
        for name in [kernel.name.clone(), format!("enqueue_{}", kernel.name)] {
            if !methods.insert(name.clone()) {
                return Err(Error(format!(
                    "kernel names collide in Rust bindings: {name}"
                )));
            }
        }
    }
    let types = HostTypes::new(abi)?;
    let mut output = String::new();
    types.emit(&mut output);
    output.push_str("pub struct Kernels<'a> {\n    device: &'a metal_oxide::Device,\n");
    for index in 0..abi.kernels.len() {
        writeln!(output, "    pipeline_{index}: metal_oxide::Pipeline,").unwrap();
    }
    output.push_str("}\n\npub fn load(device: &metal_oxide::Device, directory: impl std::convert::AsRef<std::path::Path>) -> metal_oxide::Result<Kernels<'_>> {\n    let module = metal_oxide::Module::from_artifact(device, directory)?;\n");
    writeln!(
        output,
        "    module.verify_abi({:?})?;",
        metal_oxide_artifact::sha256(abi.to_json()?.as_bytes())
    )
    .unwrap();
    output.push_str("    Ok(Kernels {\n        device,\n");
    for (index, kernel) in abi.kernels.iter().enumerate() {
        writeln!(
            output,
            "        pipeline_{index}: metal_oxide::Pipeline::new(device, &module, {:?})?,",
            kernel.name
        )
        .unwrap();
    }
    output.push_str("    })\n}\n\n#[allow(dead_code)]\nimpl Kernels<'_> {\n");
    for (index, kernel) in abi.kernels.iter().enumerate() {
        for enqueue in [false, true] {
            emit_method(&mut output, &types, index, kernel, enqueue)?;
        }
    }
    output.push_str("}\n");
    Ok(output)
}

fn emit_method(
    output: &mut String,
    types: &HostTypes,
    index: usize,
    kernel: &Kernel,
    enqueue: bool,
) -> Result<(), Error> {
    if ["self", "Self", "crate", "super", "_"].contains(&kernel.name.as_str()) {
        return Err(Error(
            "kernel name cannot be represented in Rust bindings".into(),
        ));
    }
    output.push_str("    /// # Safety\n    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.\n    #[allow(clippy::too_many_arguments)]\n");
    let config = match kernel.required_block {
        Some([x, y, z]) => format!("metal_oxide::LaunchConfig<{x}, {y}, {z}>"),
        None => "impl std::convert::Into<metal_oxide::DynamicLaunchConfig>".into(),
    };
    let name = if enqueue {
        format!("enqueue_{}", kernel.name)
    } else {
        kernel.name.clone()
    };
    write!(output, "    pub unsafe fn r#{name}(&self").unwrap();
    if enqueue {
        output.push_str(", batch: &mut metal_oxide::Batch<'_>");
    }
    write!(output, ", config: {config}").unwrap();
    let mut names = HashSet::from([
        "self".to_owned(),
        "Self".to_owned(),
        "crate".to_owned(),
        "super".to_owned(),
        "config".to_owned(),
        "batch".to_owned(),
        "_".to_owned(),
    ]);
    let mut arguments = Vec::new();
    for parameter in &kernel.parameters {
        let mut name = parameter.name.clone();
        while !names.insert(name.clone()) {
            name = format!("arg_{}_{name}", parameter.binding);
        }
        let (ty, argument) = match &parameter.ty {
            ParameterType::Value { layout } => (
                types.name(layout),
                format!("metal_oxide::Argument::value(r#{name})?"),
            ),
            ParameterType::Buffer {
                element, access, ..
            } => {
                let (borrow, method) = match access {
                    Access::Read => ("&", "read"),
                    Access::Write => ("&mut ", "write"),
                    Access::Atomic => ("&mut ", "atomic"),
                };
                (
                    format!("{borrow}metal_oxide::Buffer<{}>", types.name(element)),
                    format!("metal_oxide::Argument::{method}(r#{name})"),
                )
            }
        };
        write!(output, ", r#{name}: {ty}").unwrap();
        arguments.push(argument);
    }
    output.push_str(") -> metal_oxide::Result<()> {\n        // SAFETY: kernel-specific preconditions are delegated to this function's caller.\n        unsafe {\n");
    let receiver = if enqueue { "batch" } else { "self.device" };
    writeln!(
        output,
        "            {receiver}.launch(&self.pipeline_{index}, config, &[{}])",
        arguments.join(", ")
    )
    .unwrap();
    output.push_str("        }\n    }\n\n");
    Ok(())
}
