use super::*;
use metal_oxide::Error;

/// The safe `vec_add(a, b, out)` that the vec-add example launches.
fn vec_add() -> std::path::PathBuf {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/safe_kernel.rs",
        &[],
    );
    support::checked(output);
    directory
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn artifacts_check_arguments_before_an_empty_launch() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = support::artifact::module(&device, &vec_add())?;
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
            Launch::new(Ok(&pipeline), config, (), |()| Ok([])).wait(),
            Err(Error::Artifact(_))
        ));
        assert!(matches!(
            Launch::new(
                Ok(&pipeline),
                config,
                (&wrong, &b, &mut out),
                |(wrong, b, out)| Ok([
                    Argument::read(wrong),
                    Argument::read(b),
                    Argument::write(out)
                ])
            )
            .wait(),
            Err(Error::Artifact(_))
        ));
        assert!(matches!(
            Launch::new(Ok(&pipeline), config, (&a, &b, &out), |(a, b, out)| Ok([
                Argument::read(a),
                Argument::read(b),
                Argument::read(out)
            ]))
            .wait(),
            Err(Error::Artifact(_))
        ));
        assert!(matches!(
            Launch::new(
                Ok(&pipeline),
                config,
                (&a, &b, &mut out, 0),
                |(a, b, out, n)| Ok([
                    Argument::read(a),
                    Argument::read(b),
                    Argument::write(out),
                    Argument::value::<u32>(*n)?
                ])
            )
            .wait(),
            Err(Error::Artifact(_))
        ));
        Launch::new(Ok(&pipeline), config, (&a, &b, &mut out), |(a, b, out)| {
            Ok([Argument::read(a), Argument::read(b), Argument::write(out)])
        })
        .wait()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn artifacts_reject_modified_libraries_versions_abis_and_block_shapes() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (manifest, library) = support::artifact::manifest(&vec_add())?;
    let json = manifest.to_json()?;
    let abi = metal_oxide_artifact::sha256(manifest.abi.to_json()?.as_bytes());
    assert!(matches!(
        Module::from_bytes(&device, &json, b"modified library"),
        Err(Error::Artifact(_))
    ));
    assert!(matches!(
        Module::from_bytes(
            &device,
            &json.replace("\"version\": 4", "\"version\": 999"),
            &library
        ),
        Err(Error::Artifact(_))
    ));
    let fixed = json.replace("\"required_block\": null", "\"required_block\": [32, 1, 1]");
    let module = Module::from_bytes(&device, &fixed, &library)?;
    assert!(matches!(module.verify_abi(&abi), Err(Error::Artifact(_))));
    let pipeline = Pipeline::new(&device, &module, "vec_add")?;
    let a = device.buffer_zeroed::<f32>(0)?;
    let b = device.buffer_zeroed::<f32>(0)?;
    let mut out = device.buffer_zeroed::<f32>(0)?;
    // SAFETY: the empty grid executes no shader invocation.
    let result = unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<256>::for_elements(0)?,
            (&a, &b, &mut out),
            |(a, b, out)| Ok([Argument::read(a), Argument::read(b), Argument::write(out)]),
        )
        .wait()
    };
    assert!(matches!(result, Err(Error::InvalidLaunch(_))));
    Ok(())
}
