#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod kernels {
    include!(env!("METAL_OXIDE_BINDINGS"));
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use metal_oxide::{Device, LaunchConfig};

    let device = Device::system_default()?;
    let kernels = kernels::load(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
    println!("Device: {}", device.name());

    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        let a = (0..n).map(|i| (i % 1024) as f32 * 0.25).collect::<Vec<_>>();
        let b = (0..n).map(|i| (i % 511) as f32 * -0.5).collect::<Vec<_>>();
        let input_a = device.buffer_from_slice(&a)?;
        let input_b = device.buffer_from_slice(&b)?;
        let mut output = device.buffer_zeroed::<f32>(n as usize)?;
        let config = LaunchConfig::<256>::for_elements(n)?;

        // SAFETY: the vec_add ABI uses distinct n-element buffers and one writer per index.
        unsafe {
            kernels.vec_add(config, &input_a, &input_b, &mut output, n)?;
        }

        for (i, actual) in output.as_slice().iter().enumerate() {
            if *actual != a[i] + b[i] {
                return Err(format!("vec_add mismatch at n={n}, index={i}").into());
            }
        }
        println!(
            "vec_add n={n}: verified ({} blocks x {} threads)",
            config.grid.x,
            LaunchConfig::<256>::BLOCK.x
        );
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact)))]
fn main() -> std::process::ExitCode {
    eprintln!("run with cargo metal run -p vec-add on macOS Apple Silicon");
    std::process::ExitCode::FAILURE
}

#[cfg(all(
    test,
    target_os = "macos",
    target_arch = "aarch64",
    metal_oxide_artifact
))]
mod tests {
    use metal_oxide::{Argument, Device, Error, LaunchConfig, Module, Pipeline};

    #[test]
    #[ignore = "requires a Metal device and generated artifact"]
    fn generated_vec_add_matches_cpu() {
        super::main().unwrap();
    }

    #[test]
    #[ignore = "requires a Metal device and generated artifact"]
    fn artifact_checks_arguments_before_an_empty_launch() -> metal_oxide::Result<()> {
        let device = Device::system_default()?;
        let module = Module::from_artifact(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
        let pipeline = Pipeline::new(&device, &module, "vec_add")?;
        let a = device.buffer_zeroed::<f32>(0)?;
        let b = device.buffer_zeroed::<f32>(0)?;
        let wrong = device.buffer_zeroed::<u32>(0)?;
        let mut out = device.buffer_zeroed::<f32>(0)?;
        let config = LaunchConfig::<256>::for_elements(0)?;
        assert!(Pipeline::new(&device, &module, "missing").is_err());

        // SAFETY: every launch has an empty grid and executes no shader invocation.
        unsafe {
            assert!(matches!(
                device.launch(&pipeline, config, &[]),
                Err(Error::Artifact(_))
            ));
            assert!(matches!(
                device.launch(
                    &pipeline,
                    config,
                    &[
                        Argument::read(&wrong),
                        Argument::read(&b),
                        Argument::write(&mut out),
                        Argument::value::<u32>(0)?,
                    ]
                ),
                Err(Error::Artifact(_))
            ));
            assert!(matches!(
                device.launch(
                    &pipeline,
                    config,
                    &[
                        Argument::read(&a),
                        Argument::read(&b),
                        Argument::read(&out),
                        Argument::value::<u32>(0)?,
                    ]
                ),
                Err(Error::Artifact(_))
            ));
            assert!(matches!(
                device.launch(
                    &pipeline,
                    config,
                    &[
                        Argument::read(&a),
                        Argument::read(&b),
                        Argument::write(&mut out),
                        Argument::value::<f32>(0.0)?,
                    ]
                ),
                Err(Error::Artifact(_))
            ));
            device.launch(
                &pipeline,
                config,
                &[
                    Argument::read(&a),
                    Argument::read(&b),
                    Argument::write(&mut out),
                    Argument::value::<u32>(0)?,
                ],
            )?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a Metal device and generated artifact"]
    fn artifact_rejects_modified_library_abi_and_block_shape()
    -> Result<(), Box<dyn std::error::Error>> {
        struct Directory(std::path::PathBuf);
        impl Drop for Directory {
            fn drop(&mut self) {
                std::fs::remove_dir_all(&self.0).unwrap();
            }
        }
        let device = Device::system_default()?;
        let source = std::path::Path::new(env!("METAL_OXIDE_ARTIFACT_DIR"));
        let directory =
            Directory(std::env::temp_dir().join(format!("oxide-artifact-{}", std::process::id())));
        std::fs::create_dir(&directory.0)?;
        let manifest = std::fs::read_to_string(source.join("manifest.json"))?;
        let library = std::fs::read(source.join("kernels.metallib"))?;
        std::fs::write(directory.0.join("manifest.json"), &manifest)?;
        std::fs::write(directory.0.join("kernels.metallib"), b"modified library")?;
        assert!(matches!(
            Module::from_artifact(&device, &directory.0),
            Err(Error::Artifact(_))
        ));

        std::fs::write(directory.0.join("kernels.metallib"), library)?;
        std::fs::write(
            directory.0.join("manifest.json"),
            manifest.replace("\"version\": 3", "\"version\": 999"),
        )?;
        assert!(matches!(
            Module::from_artifact(&device, &directory.0),
            Err(Error::Artifact(_))
        ));

        std::fs::write(
            directory.0.join("manifest.json"),
            manifest.replace("\"required_block\": null", "\"required_block\": [32, 1, 1]"),
        )?;
        let module = Module::from_artifact(&device, &directory.0)?;
        assert!(matches!(
            super::kernels::load(&device, &directory.0),
            Err(Error::Artifact(_))
        ));
        let pipeline = Pipeline::new(&device, &module, "vec_add")?;
        let a = device.buffer_zeroed::<f32>(0)?;
        let b = device.buffer_zeroed::<f32>(0)?;
        let mut out = device.buffer_zeroed::<f32>(0)?;
        // SAFETY: the empty grid executes no shader invocation.
        let result = unsafe {
            device.launch(
                &pipeline,
                LaunchConfig::<256>::for_elements(0)?,
                &[
                    Argument::read(&a),
                    Argument::read(&b),
                    Argument::write(&mut out),
                    Argument::value::<u32>(0)?,
                ],
            )
        };
        assert!(matches!(result, Err(Error::InvalidLaunch(_))));
        Ok(())
    }
}
