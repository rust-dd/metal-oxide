use metal_oxide_artifact::{ArtifactFile, COMPILER_OUTPUTS};
use std::path::Path;

pub(crate) fn prepare(directory: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(directory)?;
    for file in COMPILER_OUTPUTS {
        match std::fs::remove_file(directory.join(file.name())) {
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
    entries: &[crate::collect::Entry<'tcx>],
    arguments: &[String],
    math_mode: metal_oxide_artifact::MathMode,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(
        directory.join(ArtifactFile::OxideIr.name()),
        format!("{module:?}\n"),
    )?;
    let codegen = metal_oxide_codegen::Codegen::new(module)?.with_math_mode(math_mode);
    let msl = codegen.emit()?;
    let mut abi = codegen.abi()?;
    for kernel in &mut abi.kernels {
        let entry = entries
            .iter()
            .find(|entry| entry.name == kernel.name)
            .unwrap()
            .instance;
        kernel.safe = tcx.fn_sig(entry.def_id()).skip_binder().safety().is_safe();
        let body = tcx.instance_mir(entry.def);
        let mut used = std::collections::HashSet::new();
        for (index, parameter) in kernel.parameters.iter_mut().enumerate() {
            let name = crate::collect::parameter_name(body, index).unwrap_or_default();
            if name != "_" && !name.is_empty() && name.is_ascii() && used.insert(name.clone()) {
                parameter.name = name.to_owned();
            } else {
                while !used.insert(parameter.name.clone()) {
                    parameter.name.push('_');
                }
            }
        }
    }
    abi.validate()?;
    let abi = abi.to_json()?;
    std::fs::write(directory.join(ArtifactFile::Msl.name()), msl)?;
    std::fs::write(directory.join(ArtifactFile::Abi.name()), abi)?;
    std::fs::write(
        directory.join(ArtifactFile::Checks.name()),
        serde_json::to_vec_pretty(codegen.checks())?,
    )?;
    std::fs::write(
        directory.join(ArtifactFile::RustcArgs.name()),
        serde_json::to_vec_pretty(arguments)?,
    )?;
    Ok(())
}
