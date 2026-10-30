use super::*;

fn violated(result: metal_oxide::Result<impl Sized>) -> String {
    match result {
        Err(metal_oxide::Error::Contract { kernel, relation }) => format!("{kernel}: {relation}"),
        Err(error) => panic!("expected a contract error, got {error}"),
        Ok(_) => panic!("expected a contract error"),
    }
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn launches_check_kernel_requirements_before_encoding() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/requires.rs",
        &[],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;

    let pipeline = Pipeline::new(&device, &module, "vec_add")?;
    let a = device.buffer_from_slice(&[1.0_f32, 2.0, 3.0])?;
    let b = device.buffer_from_slice(&[10.0_f32, 20.0, 30.0])?;
    let short = device.buffer_from_slice(&[10.0_f32, 20.0])?;
    let mut out = device.buffer_zeroed::<f32>(3)?;
    let config = LaunchConfig::<64>::for_elements(3)?;
    let add = |b: &metal_oxide::Buffer<f32>, out: &mut metal_oxide::Buffer<f32>| {
        // SAFETY: the kernel's requirements are checked by the launch.
        unsafe {
            Launch::new(Ok(&pipeline), config, (&a, b, out), |(a, b, out)| {
                Ok([Argument::read(a), Argument::read(b), Argument::write(out)])
            })
            .wait()
            .map(drop)
        }
    };
    assert_eq!(
        violated(add(&short, &mut out)),
        "vec_add: b.len() == out.len()"
    );
    assert_eq!(out.read().wait(), [0.0; 3]);
    add(&b, &mut out)?;
    assert_eq!(out.read().wait(), [11.0, 22.0, 33.0]);

    let pipeline = Pipeline::new(&device, &module, "pairs")?;
    let values = (0..9).map(|i| i as f32).collect::<Vec<_>>();
    let pairs = |values: &[f32], n: u32| {
        let values = device.buffer_from_slice(values)?;
        let mut out = device.buffer_zeroed::<f32>(64)?;
        // SAFETY: the kernel's requirements are checked by the launch.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<64>::new(Dim3::x(1)),
                (&values, &mut out, n),
                |(values, out, n)| {
                    Ok([
                        Argument::read(values),
                        Argument::write(out),
                        Argument::value(*n)?,
                    ])
                },
            )
            .wait()?;
        }
        Ok::<_, metal_oxide::Error>(out.read().wait().to_vec())
    };
    assert_eq!(violated(pairs(&values, 0)), "pairs: n > 0");
    assert_eq!(
        violated(pairs(&values[..8], 4)),
        "pairs: 2 * n + 1 <= values.len()"
    );
    let expected = (0..64)
        .map(|i| values[2 * (i % 4)] + values[2 * (i % 4) + 1])
        .collect::<Vec<_>>();
    assert_eq!(pairs(&values, 4)?, expected);

    let pipeline = Pipeline::new(&device, &module, "clamp_below")?;
    let values = device.buffer_from_slice(&[-9_i64, -5, 0, i64::MAX])?;
    let clamp = |floor: i64| {
        let mut out = device.buffer_zeroed::<i64>(4)?;
        // SAFETY: the kernel's requirements are checked by the launch.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<64>::for_elements(4)?,
                (&values, &mut out, floor),
                |(values, out, floor)| {
                    Ok([
                        Argument::read(values),
                        Argument::write(out),
                        Argument::value(*floor)?,
                    ])
                },
            )
            .wait()?;
        }
        Ok::<_, metal_oxide::Error>(out.read().wait().to_vec())
    };
    // A negative floor satisfies the relation only when launches read it as signed.
    assert_eq!(clamp(-5)?, [-5, -5, 0, i64::MAX]);
    assert_eq!(violated(clamp(1)), "clamp_below: floor <= 0");
    Ok(())
}
