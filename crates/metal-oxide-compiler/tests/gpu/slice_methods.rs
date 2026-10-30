use super::*;
use metal_oxide::{Error, GpuValue};

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/slice_methods.rs";

fn rearranged((values, i, j): ([u32; 8], u32, u32)) -> [u32; 8] {
    let mut output = values;
    output.swap(i as usize % 8, j as usize % 8);
    output[1..7].reverse();
    let mut local = output;
    local[..i as usize % 9].reverse();
    let (head, tail) = local.split_at_mut(4);
    core::mem::swap(&mut head[j as usize % 4], &mut tail[3]);
    let same = &raw mut local[i as usize % 8];
    // SAFETY: `same` points to an element of `local`, valid for reads and writes.
    unsafe { core::ptr::swap(same, same) };
    local
}

/// The outputs of one thread per case.
fn run<I: GpuValue, O: GpuValue>(
    device: &Device,
    entry: &str,
    cases: &[I],
) -> metal_oxide::Result<Vec<O>> {
    let pipeline = pipeline(device, FIXTURE, entry)?;
    let input = device.buffer_from_slice(cases)?;
    let mut output = device.buffer_zeroed::<O>(cases.len())?;
    // SAFETY: the safe kernel bounds its writes and checks the input length.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<64>::for_elements(cases.len() as u32)?,
            (&input, &mut output),
            |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
        )
    }
    .wait()?;
    Ok(output.read().wait().to_vec())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn swaps_and_reversals_match_rust_in_device_and_thread_memory() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let cases = (0..243_u32)
        .map(|case| {
            let values = core::array::from_fn::<u32, 8, _>(|k| case * 16 + k as u32);
            (values, case % 9, case / 9)
        })
        .collect::<Vec<_>>();
    let output = run::<_, [u32; 8]>(&device, "rearranged", &cases)?;
    for (&case, actual) in cases.iter().zip(&output) {
        assert_eq!(*actual, rearranged(case), "{case:?}");
    }
    Ok(())
}

fn copied((values, a, b): ([u32; 8], u32, u32)) -> Option<[u32; 8]> {
    let mut output = [7; 8];
    output[..4].copy_from_slice(&values[4..]);
    let (a, b) = (a as usize % 9, b as usize % 9);
    (a == b).then(|| {
        output[a..].copy_from_slice(&values[b..]);
        output
    })
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn copy_from_slice_matches_rust_and_fails_on_a_length_mismatch() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let cases = (0..18_u32)
        .map(|case| {
            let values = core::array::from_fn::<u32, 8, _>(|k| case * 16 + k as u32);
            (values, case, case % 9)
        })
        .collect::<Vec<_>>();
    let output = run::<_, [u32; 8]>(&device, "copied", &cases)?;
    for (&case, actual) in cases.iter().zip(&output) {
        assert_eq!(Some(*actual), copied(case), "{case:?}");
    }
    let mismatch = ([1; 8], 2, 3);
    assert_eq!(copied(mismatch), None);
    let Err(Error::KernelFailed(failure)) = run::<_, [u32; 8]>(&device, "copied", &[mismatch])
    else {
        panic!("expected a length mismatch failure");
    };
    assert_eq!(
        (failure.message.as_str(), failure.line),
        (
            "copy_from_slice: source slice length does not match destination slice length",
            32
        ),
        "{failure}"
    );
    assert!(
        failure.file.ends_with("fixtures/slice_methods.rs"),
        "{failure}"
    );
    Ok(())
}

fn compared((a, b, n): ([u32; 4], [u32; 4], u32)) -> [u32; 7] {
    let n = n as usize % 5;
    [
        (a == b) as u32,
        (a != b) as u32,
        (a[..n] == b[..n]) as u32,
        (a[n..] == b[..]) as u32,
        a.starts_with(&b[..n]) as u32,
        ([a[0] as u8, a[1] as u8] == [b[0] as u8, b[1] as u8]) as u32,
        ([a[2] > 1, a[3] > 1] == [b[2] > 1, b[3] > 1]) as u32,
    ]
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn array_and_slice_equality_match_rust() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let base = [1, 2, 3, 0x100];
    let mut cases = Vec::new();
    for n in 0..5 {
        cases.push((base, base, n));
        cases.push((base, [0; 4], n));
        for position in 0..4 {
            let mut other = base;
            other[position] ^= 0x100;
            cases.push((base, other, n));
        }
    }
    let output = run::<_, [u32; 7]>(&device, "compared", &cases)?;
    for (&case, actual) in cases.iter().zip(&output) {
        assert_eq!(*actual, compared(case), "{case:?}");
    }
    Ok(())
}

fn patterns((values, n): ([u32; 6], u32)) -> [u32; 8] {
    let n = n as usize % 7;
    let slice = &values[..n];
    let device = &values[n..];
    let mut local = values;
    if let Some(first) = local[..n].first_mut() {
        *first += 100;
    }
    [
        slice.first().map_or(99, |x| *x),
        slice.last().map_or(99, |x| *x),
        slice
            .split_first()
            .map_or(99, |(x, rest)| x * 10 + rest.len() as u32),
        slice
            .split_last()
            .map_or(99, |(x, rest)| x * 10 + rest.len() as u32),
        match slice {
            [] => 1,
            [x] => *x,
            [x, y] => x + y,
            [x, .., y] => x * y,
        },
        match device {
            [a, b, rest @ ..] => a + b + rest.len() as u32,
            [a] => *a,
            [] => 0,
        },
        match device {
            [.., x, y] => x ^ y,
            _ => 0,
        },
        local[0],
    ]
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn slice_patterns_and_their_core_methods_match_rust() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let cases = (0..21_u32)
        .map(|case| {
            (
                core::array::from_fn::<u32, 6, _>(|k| case * 16 + k as u32 + 1),
                case,
            )
        })
        .collect::<Vec<_>>();
    let output = run::<_, [u32; 8]>(&device, "patterns", &cases)?;
    for (&case, actual) in cases.iter().zip(&output) {
        assert_eq!(*actual, patterns(case), "{case:?}");
    }
    Ok(())
}
