use crate::support;
use metal_oxide::{Argument, Device, Dim3, Launch, LaunchConfig, Module, Operation, Pipeline};

fn module(device: &Device) -> metal_oxide::Result<Module> {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/vector_basic.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    support::artifact::module(device, &directory)
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn vectors_keep_array_borrows_and_mask_lanes() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = module(&device)?;
    let floats = run::<_, [f32; 4]>(&device, &module, "vector_f32", &[[1.0, 2.0, 3.0, -8.0]])?;
    assert_eq!(floats, [[6.0, 0.0, 0.5, 12.0]]);
    assert_eq!(
        floats,
        run::<_, [f32; 4]>(&device, &module, "scalar_f32", &[[1.0, 2.0, 3.0, -8.0]])?
    );
    for values in [[i32::MIN, 8, i32::MAX], [-3, -2, -1], [10, 20, 30]] {
        let actual = run::<_, [i32; 3]>(&device, &module, "vector_i32", &[values])?;
        let mut expected = values;
        expected[1] = expected[0];
        assert_eq!(
            actual,
            [expected.map(|n| n.saturating_add(2).max(n.wrapping_neg()) / 2)]
        );
    }
    for values in [[0_u8, 255], [127, 128], [1, 254]] {
        let actual = run::<_, u64>(&device, &module, "vector_mask", &[values])?;
        let expected = values.map(|n| {
            if n == 0 {
                n.reverse_bits()
            } else {
                n.count_ones() as u8
            }
        });
        assert_eq!(actual, [u64::from(expected[0].wrapping_add(expected[1]))]);
    }
    Ok(())
}

fn run<I: metal_oxide::GpuValue, O: metal_oxide::GpuValue>(
    device: &Device,
    module: &Module,
    name: &str,
    values: &[I],
) -> metal_oxide::Result<Vec<O>> {
    let pipeline = Pipeline::new(device, module, name)?;
    let input = device.buffer_from_slice(values)?;
    let mut output = device.buffer_zeroed::<O>(1)?;
    // SAFETY: one thread reads the supplied fixture inputs and writes its single result.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<1>::new(Dim3::x(1)),
            (&input, &mut output),
            |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
        )
    }
    .wait()?;
    Ok(output.read().wait().to_vec())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn vector_division_checks_zero_and_wraps_minimum_overflow() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = module(&device)?;
    let pipeline = Pipeline::new(&device, &module, "vector_division")?;
    let lhs = [i32::MIN, 7, -7, i32::MAX];
    let mut input = device.buffer_from_slice(&[lhs, [-1, 2, -2, 1]])?;
    let mut output = device.buffer_zeroed::<[i32; 4]>(1)?;
    for remainder in [false, true] {
        for zero_lane in 0..=4 {
            let mut rhs = [-1, 2, -2, 1];
            if zero_lane < 4 {
                rhs[zero_lane] = 0;
            }
            input.write().wait()[1] = rhs;
            output.write().wait()[0] = [123; 4];
            // SAFETY: one thread accesses two input arrays and one output array.
            let result = unsafe {
                Launch::new(
                    Ok(&pipeline),
                    LaunchConfig::<1>::new(Dim3::x(1)),
                    (&input, &mut output, remainder),
                    |(input, output, remainder)| {
                        Ok([
                            Argument::read(input),
                            Argument::write(output),
                            Argument::value(u32::from(*remainder))?,
                        ])
                    },
                )
            }
            .wait()
            .map(|_| ());
            if zero_lane < 4 {
                let Err(metal_oxide::Error::KernelFailed(failure)) = result else {
                    panic!("{result:?}")
                };
                assert!(failure.message.contains("zero"), "{failure}");
                assert_eq!(output.read().wait(), [[123; 4]]);
            } else {
                result?;
                let expected = std::array::from_fn(|i| {
                    if remainder {
                        lhs[i].wrapping_rem(rhs[i])
                    } else {
                        lhs[i].wrapping_div(rhs[i])
                    }
                });
                assert_eq!(output.read().wait(), [expected]);
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn vector_lane_borrows_check_dynamic_indices() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = module(&device)?;
    let pipeline = Pipeline::new(&device, &module, "vector_index")?;
    let input = device.buffer_from_slice(&[[i32::MAX, 20, 30]])?;
    let mut output = device.buffer_zeroed::<[i32; 3]>(1)?;
    for index in [0_usize, 1, 2, 3, usize::MAX] {
        output.write().wait()[0] = [123; 3];
        // SAFETY: one thread accesses one input and output array; lane access is checked.
        let result = unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<1>::new(Dim3::x(1)),
                (&input, &mut output, index),
                |(input, output, index)| {
                    Ok([
                        Argument::read(input),
                        Argument::write(output),
                        Argument::value(*index as u64)?,
                    ])
                },
            )
        }
        .wait()
        .map(|_| ());
        if index < 3 {
            result?;
            let mut expected = [i32::MAX, 20, 30];
            expected[index] = expected[index].wrapping_add(1);
            assert_eq!(output.read().wait(), [expected]);
        } else {
            let Err(metal_oxide::Error::KernelFailed(failure)) = result else {
                panic!("{result:?}")
            };
            assert_eq!(failure.message, "index out of bounds");
            assert_eq!(output.read().wait(), [[123; 3]]);
        }
    }
    Ok(())
}
