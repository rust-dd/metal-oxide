use crate::{
    cli::{Emit, Inspect},
    run::cargo,
};
use metal_oxide_artifact::{ArtifactFile, Manifest};
use std::{collections::BTreeSet, path::Path};

/// Builds the packages and prints the kernels their builds compiled. Each kernel macro of the
/// build lists its artifact in the file that `METAL_OXIDE_INSPECT` names.
pub(crate) fn execute(options: &Inspect) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // The macros track this variable, so a new path makes Cargo rebuild the crates whose
    // kernels report to it.
    let list = std::env::temp_dir().join(format!("metal-oxide-inspect-{}", std::process::id()));
    let _ = std::fs::remove_file(&list);
    let status = cargo()
        .arg("build")
        .args(&options.arguments)
        .env("METAL_OXIDE_INSPECT", &list)
        .status()?;
    let built = std::fs::read_to_string(&list).unwrap_or_default();
    let _ = std::fs::remove_file(&list);
    if !status.success() {
        return Err(format!("cargo build exited with {status}").into());
    }
    let package = options
        .arguments
        .iter()
        .position(|argument| argument == "-p" || argument == "--package")
        .and_then(|index| options.arguments.get(index + 1));
    let mut printed = BTreeSet::new();
    for line in built.lines() {
        let [directory, name, item] = line.splitn(3, '\t').collect::<Vec<_>>()[..] else {
            continue;
        };
        if package.is_some_and(|package| package != name) || printed.contains(directory) {
            continue;
        }
        let path = Path::new(directory);
        let manifest = Manifest::from_json(&std::fs::read_to_string(
            path.join(ArtifactFile::Manifest.name()),
        )?)?;
        let kernels = manifest
            .abi
            .kernels
            .iter()
            .map(|kernel| kernel.name.as_str())
            .collect::<Vec<_>>();
        if let Some(kernel) = &options.kernel
            && !kernels.contains(&kernel.as_str())
        {
            continue;
        }
        printed.insert(directory);
        let text = match options.emit {
            Emit::Ir => std::fs::read_to_string(path.join(ArtifactFile::OxideIr.name()))?,
            Emit::Msl => std::fs::read_to_string(path.join(ArtifactFile::Msl.name()))?,
            Emit::Air => metal_oxide_build::disassemble(&path.join(ArtifactFile::Air.name()))?,
        };
        println!("// {name}::{item}: {}", kernels.join(", "));
        print!("{text}");
    }
    if printed.is_empty() {
        return Err(match &options.kernel {
            Some(kernel) => format!("the build compiled no kernel named {kernel}"),
            None => "the build compiled no kernels".into(),
        }
        .into());
    }
    Ok(())
}
