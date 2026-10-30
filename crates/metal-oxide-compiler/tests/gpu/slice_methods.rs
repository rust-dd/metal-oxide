use super::*;
use metal_oxide::GpuValue;

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
