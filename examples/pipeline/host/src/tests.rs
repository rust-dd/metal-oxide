use metal_oxide::{Argument, Device, Error, LaunchConfig, Module, Pipeline};

#[test]
#[ignore = "requires a Metal device and generated artifact"]
fn async_generated_kernel_chain_matches_cpu() {
    super::main().unwrap();
}

#[test]
#[ignore = "requires a Metal device and generated artifact"]
fn cancelled_generated_submission_keeps_buffers_synchronized() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let kernels = super::kernels::load(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
    let n = 1_000_003;
    let input = device.buffer_from_slice(&vec![2.0_f32; n])?;
    let mut output = device.buffer_zeroed::<f32>(n)?;
    // SAFETY: the specialized grid has one writer per valid index and disjoint n-element buffers.
    let submission = unsafe {
        device.submit(|batch| {
            kernels.enqueue_scale_128(
                batch,
                LaunchConfig::<128>::for_elements(n as u32)?,
                &input,
                &mut output,
                n as u32,
            )
        })
    }?;
    drop(submission);
    assert!(output.as_slice().iter().all(|&value| value == 7.0));
    Ok(())
}

#[test]
#[ignore = "requires a Metal device and generated artifact"]
fn raw_launch_rejects_the_wrong_const_specialization() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_artifact(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
    let pipeline = Pipeline::new(&device, &module, "scale_128")?;
    let input = device.buffer_zeroed::<f32>(0)?;
    let mut output = device.buffer_zeroed::<f32>(0)?;
    // SAFETY: the empty grid executes no invocation; block validation must still reject the mismatch.
    let result = unsafe {
        device.launch(
            &pipeline,
            LaunchConfig::<256>::for_elements(0)?,
            &[
                Argument::read(&input),
                Argument::write(&mut output),
                Argument::value::<u32>(0)?,
            ],
        )
    };
    assert!(matches!(result, Err(Error::InvalidLaunch(_))));
    Ok(())
}
