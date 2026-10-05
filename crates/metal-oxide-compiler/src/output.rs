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
    for name in ["kernels.oxide-ir", "kernels.metal"] {
        match std::fs::remove_file(directory.join(name)) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

pub(crate) fn write(
    directory: &Path,
    module: &metal_oxide_ir::Module,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(directory.join("kernels.oxide-ir"), format!("{module:#?}\n"))?;
    let msl = metal_oxide_codegen::emit(module)?;
    std::fs::write(directory.join("kernels.metal"), msl)?;
    Ok(())
}
