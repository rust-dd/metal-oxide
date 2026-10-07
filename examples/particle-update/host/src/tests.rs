use super::{kernels, run};
use metal_oxide::{Argument, Device, Error, GpuValue, LaunchConfig, Module, Pipeline};

#[test]
#[ignore = "requires a Metal device and generated artifact"]
fn generated_particle_update_matches_cpu_and_msl() -> metal_oxide::Result<()> {
    run::verify()
}

#[test]
#[ignore = "requires a Metal device and generated artifact"]
fn generated_particle_chain_preserves_nested_fields() -> metal_oxide::Result<()> {
    run::verify_chain()
}

#[test]
#[ignore = "requires a Metal device and generated artifact"]
fn dropped_and_forgotten_generated_particle_submissions_wait_for_readback()
-> metal_oxide::Result<()> {
    run::verify_cancelled()
}

#[test]
#[ignore = "requires a Metal device and generated artifact"]
fn record_and_config_layouts_are_checked_before_dispatch() -> metal_oxide::Result<()> {
    assert_eq!(
        (kernels::Particle::SIZE, kernels::Particle::ALIGNMENT),
        (28, 4)
    );
    assert_eq!(kernels::Motion::SIZE, 16);
    assert_eq!(kernels::Config::SIZE, 12);
    let device = Device::system_default()?;
    let module = Module::from_artifact(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
    let pipeline = Pipeline::new(&device, &module, "particle_update")?;
    let input = device.buffer_zeroed::<kernels::Particle>(0)?;
    let mut output = device.buffer_zeroed::<kernels::Particle>(0)?;
    let wrong = device.buffer_zeroed::<[u32; 7]>(0)?;
    // SAFETY: empty launches exercise metadata validation without running shader code.
    unsafe {
        assert!(matches!(
            device.launch(
                &pipeline,
                LaunchConfig::<64>::for_elements(0)?,
                &[
                    Argument::read(&wrong),
                    Argument::write(&mut output),
                    Argument::value(run::config())?,
                    Argument::value(0_u32)?
                ]
            ),
            Err(Error::Artifact(_))
        ));
        assert!(matches!(
            device.launch(
                &pipeline,
                LaunchConfig::<64>::for_elements(0)?,
                &[
                    Argument::read(&input),
                    Argument::write(&mut output),
                    Argument::value(0_u32)?,
                    Argument::value(0_u32)?
                ]
            ),
            Err(Error::Artifact(_))
        ));
        assert!(matches!(
            device.launch(
                &pipeline,
                LaunchConfig::<32>::for_elements(0)?,
                &[
                    Argument::read(&input),
                    Argument::write(&mut output),
                    Argument::value(run::config())?,
                    Argument::value(0_u32)?
                ]
            ),
            Err(Error::InvalidLaunch(_))
        ));
    }
    Ok(())
}
