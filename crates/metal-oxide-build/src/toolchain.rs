use crate::process::{self, Result};
use metal_oxide_artifact::{Abi, ArtifactFile, LOGGING_FLAG};
use std::{
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

pub(crate) fn metal_flags(abi: &Abi) -> Vec<String> {
    let logging = abi.required_features.iter().any(|f| f == "logging");
    ["-std=metal3.2", "-mmacosx-version-min=15.0"]
        .into_iter()
        .chain(abi.math_mode.metal_flags())
        .chain(["-O2"])
        .chain(logging.then_some(LOGGING_FLAG))
        .map(str::to_owned)
        .collect()
}

pub(crate) struct Metal {
    program: PathBuf,
    prefix: Vec<String>,
    pub(crate) version: String,
    pub(crate) sdk: String,
    sdk_path: PathBuf,
}

impl Metal {
    pub(crate) fn find() -> Result<Self> {
        let sdk = process::capture(Command::new("xcrun").args([
            "--sdk",
            "macosx",
            "--show-sdk-version",
        ]))?;
        let sdk_path = PathBuf::from(process::capture(Command::new("xcrun").args([
            "--sdk",
            "macosx",
            "--show-sdk-path",
        ]))?);
        let prefix = vec!["-sdk".into(), "macosx".into(), "metal".into()];
        if let Ok(version) = process::capture(Command::new("xcrun").args(&prefix).arg("--version"))
        {
            return Ok(Self {
                program: "xcrun".into(),
                prefix,
                version,
                sdk,
                sdk_path,
            });
        }
        let component = process::capture(Command::new("xcodebuild").args([
            "-showComponent",
            "MetalToolchain",
            "-json",
        ]))?;
        let component: serde_json::Value = serde_json::from_str(&component)?;
        let mount = component["toolchainSearchPath"]
            .as_str()
            .ok_or("install Metal with xcodebuild -downloadComponent MetalToolchain")?;
        let directory = Path::new(mount).join("Metal.xctoolchain/usr/metal");
        let mut candidates = std::fs::read_dir(directory)?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let version = e.file_name().to_str()?.parse::<u32>().ok()?;
                Some((version, e.path().join("bin/metal")))
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|a| std::cmp::Reverse(a.0));
        for (_, program) in candidates {
            if let Ok(version) = process::capture(Command::new(&program).arg("--version")) {
                return Ok(Self {
                    program,
                    prefix: vec![],
                    version,
                    sdk,
                    sdk_path,
                });
            }
        }
        Err("Metal compiler unavailable; run xcodebuild -downloadComponent MetalToolchain".into())
    }

    /// The textual form of an AIR file, from the toolchain's `metal-opt`.
    pub(crate) fn disassemble(&self, air: &Path) -> Result<String> {
        let mut command = match self.prefix.as_slice() {
            [] => Command::new(self.program.with_file_name("metal-opt")),
            [sdk @ .., _] => {
                let mut command = Command::new(&self.program);
                command.args(sdk).arg("metal-opt");
                command
            }
        };
        command
            .env("SDKROOT", &self.sdk_path)
            .args(["-S", "-o", "-"])
            .arg(air);
        process::capture(&mut command).map(|text| {
            text + "
"
        })
    }

    pub(crate) fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.prefix).env("SDKROOT", &self.sdk_path);
        command
    }

    pub(crate) fn compile(&self, directory: &Path, abi: &Abi) -> Result<()> {
        let mut compile = self.command();
        compile
            .args(metal_flags(abi))
            .arg("-c")
            .arg(directory.join(ArtifactFile::Msl.name()))
            .arg("-o")
            .arg(directory.join(ArtifactFile::Air.name()));
        let mut link = self.command();
        link.arg(directory.join(ArtifactFile::Air.name()))
            .arg("-o")
            .arg(directory.join(ArtifactFile::Metallib.name()));
        let commands = [&compile, &link].map(|command| {
            serde_json::json!({
                "program": command.get_program().to_string_lossy(),
                "arguments": command.get_args().map(|a| a.to_string_lossy()).collect::<Vec<_>>(),
                "SDKROOT": self.sdk_path,
            })
        });
        std::fs::write(
            directory.join("metal-commands.json"),
            serde_json::to_vec_pretty(&commands)?,
        )?;
        let mut log = std::fs::File::create(directory.join("metal.log"))?;
        for command in [&mut compile, &mut link] {
            let output = command.output()?;
            log.write_all(&output.stdout)?;
            log.write_all(&output.stderr)?;
            std::io::stdout().write_all(&output.stdout)?;
            std::io::stderr().write_all(&output.stderr)?;
            if !output.status.success() {
                return Err(format!("{command:?} exited with {}", output.status).into());
            }
        }
        Ok(())
    }
}
