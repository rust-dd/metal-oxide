use std::path::{Path, PathBuf};

pub(crate) fn take_directory(arguments: &mut Vec<String>) -> Result<Option<PathBuf>, String> {
    let mut directory = None;
    let mut index = 1;
    while index < arguments.len() {
        if arguments[index] != "--metal-output" {
            index += 1;
            continue;
        }
        if directory.is_some() {
            return Err("--metal-output must be specified once".into());
        }
        let value = arguments
            .get(index + 1)
            .filter(|v| !v.starts_with('-') && !v.is_empty())
            .ok_or("--metal-output requires a directory")?;
        directory = Some(PathBuf::from(value));
        arguments.drain(index..index + 2);
    }
    Ok(directory)
}

pub(crate) fn prepare(directory: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(directory)?;
    for name in [
        "kernels.oxide-ir",
        "kernels.metal",
        "abi.json",
        "bindings.rs",
    ] {
        match std::fs::remove_file(directory.join(name)) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

pub(crate) fn write<'tcx>(
    directory: &Path,
    module: &metal_oxide_ir::Module,
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    entries: &[rustc_middle::ty::Instance<'tcx>],
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(directory.join("kernels.oxide-ir"), format!("{module:#?}\n"))?;
    let msl = metal_oxide_codegen::emit(module)?;
    std::fs::write(directory.join("kernels.metal"), msl)?;
    let mut abi = metal_oxide_codegen::abi(module)?;
    for kernel in &mut abi.kernels {
        let entry = entries
            .iter()
            .find(|e| tcx.item_name(e.def_id()).as_str() == kernel.name)
            .unwrap();
        let body = tcx.instance_mir(entry.def);
        let mut used = std::collections::HashSet::new();
        for (index, parameter) in kernel.parameters.iter_mut().enumerate() {
            let local = body.args_iter().nth(index).unwrap();
            let name = body
                .var_debug_info
                .iter()
                .find_map(|debug| match &debug.value {
                    rustc_middle::mir::VarDebugInfoContents::Place(place)
                        if place.local == local && place.projection.is_empty() =>
                    {
                        Some(debug.name.to_string())
                    }
                    _ => None,
                })
                .unwrap_or_default();
            let name = name.strip_prefix("r#").unwrap_or(&name);
            if name != "_" && !name.is_empty() && used.insert(name.to_owned()) {
                parameter.name = name.to_owned();
            } else {
                while !used.insert(parameter.name.clone()) {
                    parameter.name.push('_');
                }
            }
        }
    }
    std::fs::write(directory.join("abi.json"), abi.to_json()?)?;
    std::fs::write(
        directory.join("bindings.rs"),
        metal_oxide_codegen::bindings(&abi)?,
    )?;
    Ok(())
}
