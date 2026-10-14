use crate::cli::{Action, Options};
use metal_oxide_artifact::ArtifactFile;
use metal_oxide_build::Kernels;
use std::process::Command;

pub(crate) fn execute(options: &Options) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let host = crate::metadata::host(options)?;
    let kernels = Kernels::new(&host.kernels, crate::compiler::find()?, options.math_mode)?;
    if let Action::Inspect(emit) = options.action {
        print!("{}", kernels.emit(emit)?);
        return Ok(());
    }
    let directory = kernels.compile()?;
    let mut command = Command::new("cargo");
    command
        .current_dir(&host.workspace_root)
        .arg(match options.action {
            Action::Build => "build",
            Action::Run => "run",
            Action::Test => "test",
            Action::Inspect(_) => unreachable!(),
        })
        .args(["--package", &host.name, "--locked"])
        .arg("--manifest-path")
        .arg(&host.manifest)
        .env(
            "METAL_OXIDE_BINDINGS",
            directory.join(ArtifactFile::Bindings.name()),
        )
        .env("METAL_OXIDE_ARTIFACT_DIR", &directory);
    if options.release {
        command.arg("--release");
    }
    if options.validate {
        command
            .env("MTL_DEBUG_LAYER", "1")
            .env("MTL_SHADER_VALIDATION", "1");
    }
    if options.action == Action::Test {
        command.args(["--", "--ignored", "--test-threads=1"]);
    } else if !options.arguments.is_empty() {
        command.arg("--");
    }
    command.args(&options.arguments);
    let status = command.status()?;
    if !status.success() {
        return Err(format!("cargo exited with {status}").into());
    }
    Ok(())
}
