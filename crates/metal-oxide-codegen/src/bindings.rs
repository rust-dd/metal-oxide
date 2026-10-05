use metal_oxide_artifact::{Abi, Access, Error, ParameterType};
use std::{collections::HashSet, fmt::Write};

pub fn bindings(abi: &Abi) -> Result<String, Error> {
    abi.validate()?;
    let mut output =
        String::from("pub struct Kernels<'a> {\n    device: &'a metal_oxide::Device,\n");
    for index in 0..abi.kernels.len() {
        writeln!(output, "    pipeline_{index}: metal_oxide::Pipeline,").unwrap();
    }
    output.push_str("}\n\npub fn load(device: &metal_oxide::Device, directory: impl AsRef<std::path::Path>) -> metal_oxide::Result<Kernels<'_>> {\n    let module = metal_oxide::Module::from_artifact(device, directory)?;\n    Ok(Kernels {\n        device,\n");
    for (index, kernel) in abi.kernels.iter().enumerate() {
        writeln!(
            output,
            "        pipeline_{index}: metal_oxide::Pipeline::new(device, &module, {:?})?,",
            kernel.name
        )
        .unwrap();
    }
    output.push_str("    })\n}\n\nimpl Kernels<'_> {\n");
    for (index, kernel) in abi.kernels.iter().enumerate() {
        if ["self", "Self", "crate", "super", "_"].contains(&kernel.name.as_str()) {
            return Err(Error(
                "kernel name cannot be represented in Rust bindings".into(),
            ));
        }
        output.push_str("    /// # Safety\n    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.\n");
        write!(
            output,
            "    pub unsafe fn r#{}(&self, config: impl Into<metal_oxide::DynamicLaunchConfig>",
            kernel.name
        )
        .unwrap();
        let mut names = HashSet::from([
            "self".to_owned(),
            "Self".to_owned(),
            "crate".to_owned(),
            "super".to_owned(),
            "config".to_owned(),
            "_".to_owned(),
        ]);
        let mut arguments = Vec::new();
        for parameter in &kernel.parameters {
            let mut name = parameter.name.clone();
            while !names.insert(name.clone()) {
                name = format!("arg_{}_{name}", parameter.binding);
            }
            let (ty, argument) = match parameter.ty {
                ParameterType::Scalar { scalar } => (
                    scalar.rust_name().to_owned(),
                    format!("metal_oxide::Argument::{}(r#{name})", scalar.rust_name()),
                ),
                ParameterType::Buffer { element, access } => {
                    let (borrow, method) = match access {
                        Access::Read => ("&", "read"),
                        Access::Write => ("&mut ", "write"),
                    };
                    (
                        format!("{borrow}metal_oxide::Buffer<{}>", element.rust_name()),
                        format!("metal_oxide::Argument::{method}(r#{name})"),
                    )
                }
            };
            write!(output, ", r#{name}: {ty}").unwrap();
            arguments.push(argument);
        }
        output.push_str(") -> metal_oxide::Result<()> {\n        // SAFETY: kernel-specific preconditions are delegated to this function's caller.\n        unsafe {\n");
        writeln!(
            output,
            "            self.device.launch(&self.pipeline_{index}, config, &[{}])",
            arguments.join(", ")
        )
        .unwrap();
        output.push_str("        }\n    }\n\n");
    }
    output.push_str("}\n");
    Ok(output)
}
