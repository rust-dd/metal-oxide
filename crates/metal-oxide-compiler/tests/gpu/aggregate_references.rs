use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn tuples_and_records_of_references_match_the_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/aggregate_references.rs",
        &[],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;
    let n = 1000;
    let a = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..n).map(|i| (i % 7) as f32).collect::<Vec<_>>();
    let inputs = [device.buffer_from_slice(&a)?, device.buffer_from_slice(&b)?];
    let local = [4.0_f32, 5.0, 6.0, 7.0];
    let chosen = |i: usize| {
        if i.is_multiple_of(2) {
            a[i % 4]
        } else {
            b[i % 4]
        }
    };
    let expected = [
        (0..n)
            .map(|i| a[i] + b[i] + [0.5, 1.5, 2.5, 3.5][i % 4])
            .collect::<Vec<_>>(),
        (0..n)
            .map(|i| a[i] * a[i] + b[i] * b[i])
            .collect::<Vec<_>>(),
        (0..n)
            .map(|i| a[i] + b[i] + chosen(i) + local[i % 4])
            .collect::<Vec<_>>(),
    ];
    let kernels = ["tuple_slices", "record_slices", "reference_arrays"];
    for (kernel, expected) in kernels.into_iter().zip(expected) {
        let pipeline = Pipeline::new(&device, &module, kernel)?;
        let mut out = device.buffer_zeroed::<f32>(n)?;
        // SAFETY: the grid covers every element; each thread writes only its own.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<256>::for_elements(n as u32)?,
                (&inputs[0], &inputs[1], &mut out),
                |(a, b, out)| Ok([Argument::read(a), Argument::read(b), Argument::write(out)]),
            )
            .wait()?;
        }
        assert_eq!(out.read().wait(), expected, "{kernel}");
    }
    let pipeline = Pipeline::new(&device, &module, "indexed_tuple")?;
    let mut out = device.buffer_zeroed::<f32>(n)?;
    // SAFETY: each thread writes only its own element.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<256>::for_elements(n as u32)?,
            (&inputs[0], &mut out),
            |(a, out)| Ok([Argument::read(a), Argument::write(out)]),
        )
        .wait()?;
    }
    assert_eq!(
        out.read().wait(),
        a.iter().map(|x| x * 2.0).collect::<Vec<_>>()
    );
    Ok(())
}
