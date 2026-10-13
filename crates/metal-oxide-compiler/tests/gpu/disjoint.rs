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

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn threads_iterate_their_own_chunks_of_a_buffer_slice() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/chunk_sums.rs",
        &[],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;
    let pipeline = Pipeline::new(&device, &module, "chunk_sums")?;
    let values = (0..400).map(|i| (i % 7) as f32).collect::<Vec<_>>();
    let input = device.buffer_from_slice(&values)?;
    let mut out = device.buffer_zeroed::<f32>(100)?;
    // SAFETY: the grid covers every output element; each thread writes only its own.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<64>::for_elements(100)?,
            (&input, &mut out),
            |(input, out)| Ok([Argument::read(input), Argument::write(out)]),
        )
        .wait()?;
    }
    let expected = values
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| chunk.iter().sum::<f32>() + chunk[0] * chunk[1] + chunk[2] * chunk[3])
        .collect::<Vec<_>>();
    assert_eq!(out.read().wait(), expected);
    Ok(())
}
