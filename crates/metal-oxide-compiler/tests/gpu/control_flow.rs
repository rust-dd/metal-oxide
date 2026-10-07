use super::*;
use metal_oxide::GpuScalar;

#[path = "../fixtures/control_helpers.rs"]
mod cpu;

const REFERENCE: &str = include_str!("../../../../benchmarks/reference-msl/control_flow.metal");

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn integer_matches_agree_with_cpu_and_handwritten_msl() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let generated = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/control_match.rs",
        "classify",
    )?;
    let module = Module::from_source(&device, REFERENCE)?;
    let reference = Pipeline::new(&device, &module, "classify")?;
    let values = [
        i32::MIN,
        -65537,
        -257,
        -7,
        -3,
        -1,
        0,
        1,
        2,
        7,
        9,
        10,
        12,
        13,
        255,
        256,
        65535,
        65536,
        i32::MAX,
    ];
    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        compare(&device, &generated, &reference, &values, n, cpu::classify)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn loop_exits_agree_with_cpu_and_handwritten_msl() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/loop_exits.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let generated = Module::from_source(
        &device,
        &std::fs::read_to_string(directory.join("kernels.metal"))?,
    )?;
    let reference = Module::from_source(&device, REFERENCE)?;
    for (entry, values, expected) in [
        (
            "bounded_search",
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, u32::MAX][..],
            cpu::search as fn(u32) -> u32,
        ),
        (
            "nested_exits",
            &[0, 1, 2, 3, 4, 7, u32::MAX][..],
            cpu::nested as fn(u32) -> u32,
        ),
    ] {
        let generated = Pipeline::new(&device, &generated, entry)?;
        let reference = Pipeline::new(&device, &reference, entry)?;
        for n in [0_u32, 1, 255, 256, 257, 4099] {
            compare(&device, &generated, &reference, values, n, expected)?;
        }
    }
    Ok(())
}

fn compare<T: GpuScalar>(
    device: &Device,
    generated: &Pipeline,
    reference: &Pipeline,
    pattern: &[T],
    n: u32,
    expected: impl Fn(T) -> u32,
) -> metal_oxide::Result<()> {
    let values = (0..n as usize)
        .map(|i| pattern[i % pattern.len()])
        .collect::<Vec<_>>();
    let input = device.buffer_from_slice(&values)?;
    let expected = values.iter().copied().map(expected).collect::<Vec<_>>();
    for config in [
        DynamicLaunchConfig::from(LaunchConfig::<32>::for_elements(n)?),
        DynamicLaunchConfig::from(LaunchConfig::<256>::for_elements(n)?),
    ] {
        let sentinel = 0xdead_beef;
        let len = n as usize + config.block.x as usize;
        for pipeline in [generated, reference] {
            let mut output = device.buffer_from_slice(&vec![sentinel; len])?;
            // SAFETY: distinct buffers cover n; each active thread owns one output element.
            unsafe {
                device.launch(
                    pipeline,
                    config,
                    &[
                        Argument::read(&input),
                        Argument::write(&mut output),
                        Argument::u32(n),
                    ],
                )?;
            }
            assert_eq!(
                &output.as_slice()[..n as usize],
                expected,
                "n={n}, block={}",
                config.block.x
            );
            assert!(
                output.as_slice()[n as usize..]
                    .iter()
                    .all(|&value| value == sentinel)
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn uniform_exits_preserve_barriers_and_simd_helper_calls() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let generated = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/cooperative_control.rs",
        "cooperative_exits",
    )?;
    let module = Module::from_source(&device, REFERENCE)?;
    let reference = Pipeline::new(&device, &module, "cooperative_exits")?;
    for mode in [0_u32, 1, 2, 3, 4, 7, 9, u32::MAX] {
        let mut expected = [u32::MAX; 96];
        let mut expected_sum = [-1.0; 96];
        if mode != 9 {
            let delta = cooperative_delta(mode);
            for i in 0..96 {
                expected[i] = ((i as u32 + 1) & 31) + 1 + delta;
                expected_sum[i] = 528.0;
            }
        }
        for pipeline in [&generated, &reference] {
            let mut output = device.buffer_from_slice(&[u32::MAX; 96])?;
            let mut sums = device.buffer_from_slice(&[-1.0_f32; 96])?;
            // SAFETY: three 32-thread blocks own distinct outputs; mode is uniform across all lanes.
            unsafe {
                device.launch(
                    pipeline,
                    LaunchConfig::<32>::new(Dim3::x(3)),
                    &[
                        Argument::write(&mut output),
                        Argument::write(&mut sums),
                        Argument::u32(mode),
                    ],
                )?;
            }
            assert_eq!(output.as_slice(), expected, "mode={mode}");
            assert_eq!(sums.as_slice(), expected_sum, "mode={mode}");
        }
    }
    Ok(())
}

fn cooperative_delta(mode: u32) -> u32 {
    let mut delta = 0;
    let mut i = 0;
    'outer: while i < 4 {
        i += 1;
        let mut j = 0;
        while j < 3 {
            j += 1;
            match mode {
                0 if j == 2 => continue,
                1 if i == 2 && j == 2 => break,
                2 if i == 3 && j == 1 => continue 'outer,
                3 if i == 4 && j == 2 => break 'outer,
                _ => {}
            }
            delta += i * 10 + j;
        }
    }
    delta
}
