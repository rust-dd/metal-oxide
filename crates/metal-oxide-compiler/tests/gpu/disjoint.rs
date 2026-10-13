use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn threads_past_the_end_get_no_element() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/disjoint.rs",
        &[],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;
    let pipeline = Pipeline::new(&device, &module, "add")?;
    let a = (0..1000).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..1000).map(|i| 2.0 * i as f32).collect::<Vec<_>>();
    let a = device.buffer_from_slice(&a)?;
    let b = device.buffer_from_slice(&b)?;
    let mut out = device.buffer_zeroed::<f32>(1000)?;
    // SAFETY: the grid covers every element; each thread writes only its own.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<256>::for_elements(1000)?,
            (&a, &b, &mut out),
            |(a, b, out)| Ok([Argument::read(a), Argument::read(b), Argument::write(out)]),
        )
        .wait()?;
    }
    let out = out.read().wait();
    assert!(out.iter().enumerate().all(|(i, x)| *x == 3.0 * i as f32));
    Ok(())
}
