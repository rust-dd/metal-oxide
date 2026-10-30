use super::*;
use metal_oxide::{Error, GpuValue};

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/logarithms.rs";
const LOGARITHM: &str = "argument of integer logarithm must be positive";
const ROOT: &str = "argument of integer square root cannot be negative";

fn logarithms((a, b, c): (u32, i32, u64)) -> Option<[u32; 8]> {
    Some([
        a.checked_ilog2()?,
        a.checked_ilog10()?,
        a.checked_ilog(3)?,
        (a as u16 | 1).checked_ilog10()?,
        b.checked_ilog2()?,
        b.checked_ilog10()?,
        c.checked_ilog2()?,
        c.checked_ilog10()?,
    ])
}

fn square_roots((a, b): (i32, i64)) -> Option<[i64; 4]> {
    Some([
        a.checked_isqrt()? as i64,
        (a as i16 & i16::MAX).isqrt() as i64,
        b.checked_isqrt()?,
        a.unsigned_abs().isqrt() as i64,
    ])
}

/// Runs one thread per case and returns the launch result with every output element.
fn run<I: GpuValue, O: GpuValue>(
    device: &Device,
    entry: &str,
    cases: &[I],
    sentinel: O,
) -> metal_oxide::Result<(metal_oxide::Result<()>, Vec<O>)> {
    let pipeline = pipeline(device, FIXTURE, entry)?;
    let input = device.buffer_from_slice(cases)?;
    let mut output = device.buffer_from_slice(&vec![sentinel; cases.len()])?;
    // SAFETY: the safe kernel bounds its writes and checks the input length.
    let result = unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<64>::for_elements(cases.len() as u32)?,
            (&input, &mut output),
            |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
        )
    }
    .wait()
    .map(drop);
    Ok((result, output.read().wait().to_vec()))
}

/// Asserts that a launch failed with `message` at the caller of the panicking method.
fn failed(result: metal_oxide::Result<()>, kernel: &str, message: &str, line: u32) {
    let Err(Error::KernelFailed(failure)) = result else {
        panic!("expected {kernel} to fail with {message}, got {result:?}");
    };
    assert_eq!(
        (
            failure.kernel.as_str(),
            failure.message.as_str(),
            failure.line
        ),
        (kernel, message, line),
        "{failure}"
    );
    assert!(
        failure.file.ends_with("fixtures/logarithms.rs"),
        "{failure}"
    );
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn integer_logarithms_match_rust_and_fail_on_nonpositive_arguments() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let unsigned = [
        1,
        2,
        3,
        9,
        10,
        99,
        100,
        1000,
        65_535,
        65_536,
        1 << 31,
        u32::MAX,
    ];
    let signed = [1, 2, 9, 10, 1000, 1 << 30, i32::MAX];
    let wide = [1, 2, 10, 1 << 32, 10_u64.pow(19), u64::MAX];
    let cases = (0..unsigned.len() * signed.len() * wide.len())
        .map(|i| {
            (
                unsigned[i % unsigned.len()],
                signed[i / unsigned.len() % signed.len()],
                wide[i / (unsigned.len() * signed.len())],
            )
        })
        .collect::<Vec<_>>();
    let (result, output) = run(&device, "logarithms", &cases, [u32::MAX; 8])?;
    result?;
    for (&case, &actual) in cases.iter().zip(&output) {
        assert_eq!(Some(actual), logarithms(case), "{case:?}");
    }
    for (case, line) in [
        ((0, 1, 1), 11),
        ((1, 0, 1), 15),
        ((1, -5, 1), 15),
        ((1, 1, 0), 17),
    ] {
        assert_eq!(logarithms(case), None);
        let (result, output) = run(&device, "logarithms", &[case], [7_u32; 8])?;
        failed(result, "logarithms", LOGARITHM, line);
        assert_eq!(output, [[7; 8]]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn integer_square_roots_match_rust_and_fail_on_negative_arguments() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let signed = [
        0,
        1,
        2,
        3,
        4,
        15,
        16,
        17,
        65_535,
        65_536,
        i16::MAX as i32,
        i32::MAX,
    ];
    let wide = [0, 1, 99, 100, 1 << 32, (1 << 62) - 1, 1 << 62, i64::MAX];
    let cases = (0..signed.len() * wide.len())
        .map(|i| (signed[i % signed.len()], wide[i / signed.len()]))
        .collect::<Vec<_>>();
    let (result, output) = run(&device, "square_roots", &cases, [-1_i64; 4])?;
    result?;
    for (&case, &actual) in cases.iter().zip(&output) {
        assert_eq!(Some(actual), square_roots(case), "{case:?}");
    }
    for (case, line) in [
        ((-1, 0), 29),
        ((i32::MIN, 0), 29),
        ((0, -1), 31),
        ((0, i64::MIN), 31),
    ] {
        assert_eq!(square_roots(case), None);
        let (result, output) = run(&device, "square_roots", &[case], [7_i64; 4])?;
        failed(result, "square_roots", ROOT, line);
        assert_eq!(output, [[7; 4]]);
    }
    Ok(())
}
