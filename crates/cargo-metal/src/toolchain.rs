use crate::process::{self, Result};
use metal_oxide_artifact::ArtifactFile;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

pub(crate) const METAL_FLAGS: &[&str] = &[
    "-std=metal3.1",
    "-mmacosx-version-min=15.0",
    "-fmetal-math-mode=safe",
    "-fmetal-math-fp32-functions=precise",
    "-ffp-contract=off",
    "-O2",
];

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

    pub(crate) fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.prefix).env("SDKROOT", &self.sdk_path);
        command
    }

    pub(crate) fn compile(&self, directory: &Path) -> Result<()> {
        process::run(
            self.command()
                .args(METAL_FLAGS)
                .arg("-c")
                .arg(directory.join(ArtifactFile::Msl.name()))
                .arg("-o")
                .arg(directory.join(ArtifactFile::Air.name())),
        )?;
        process::run(
            self.command()
                .arg(directory.join(ArtifactFile::Air.name()))
                .arg("-o")
                .arg(directory.join(ArtifactFile::Metallib.name())),
        )
    }
}
