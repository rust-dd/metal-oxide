use super::*;

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/array_methods.rs";

fn iterated(values: [u32; 4]) -> [u32; 5] {
    let mut total = 0_u32;
    for value in values {
        total = total.wrapping_mul(31).wrapping_add(value);
    }
    let mut weighted = 0_u32;
    for (i, value) in values.into_iter().enumerate() {
        weighted = weighted.wrapping_add(value.wrapping_mul(i as u32 + 1));
    }
    let mut dot = 0_u32;
    for (x, y) in values.into_iter().zip([3_u32, 5, 7, 11]) {
        dot = dot.wrapping_add(x.wrapping_mul(y));
    }
    let mut nested = 0_u32;
    for row in [[values[0], values[1]], [values[2], values[3]]] {
        for value in row {
            nested = nested.wrapping_mul(7) ^ value;
        }
    }
    let mut iterator = values.into_iter();
    let head = iterator.next().unwrap_or(0) ^ iterator.next().unwrap_or(0);
    let mut rest = 0_u32;
    for value in iterator {
        rest = rest.wrapping_add(value);
    }
    [total, weighted, dot, nested, head.wrapping_mul(rest)]
}

fn built(values: [u32; 4]) -> [u32; 6] {
    let tripled = values.map(|value| value.wrapping_mul(3));
    let key = values[1];
    let keyed = values.map(|value| value ^ key);
    let mut count = 0_u32;
    let counted = values.map(|value| {
        count += 1;
        value.wrapping_add(count)
    });
    let halves = values.map(|value| value as u8 as f32 * 0.5);
    let mirrored = core::array::from_fn::<u32, 4, _>(|i| values[i].wrapping_add(values[3 - i]));
    let grid = core::array::from_fn::<[u32; 3], 2, _>(|i| {
        core::array::from_fn(|j| values[i + j].wrapping_mul(j as u32))
    });
    [
        tripled[0] ^ tripled[3],
        keyed[2],
        counted[3].wrapping_add(count),
        halves[1].to_bits() ^ halves[2].to_bits(),
        mirrored[1] ^ mirrored[3],
        grid[1][2] ^ grid[0][1],
    ]
}

/// The outputs of one thread per case of four values.
fn run<const OUTPUTS: usize>(
    device: &Device,
    entry: &str,
    cases: &[[u32; 4]],
) -> metal_oxide::Result<Vec<[u32; OUTPUTS]>> {
    let pipeline = pipeline(device, FIXTURE, entry)?;
    let input = device.buffer_from_slice(cases)?;
    let mut output = device.buffer_zeroed::<[u32; OUTPUTS]>(cases.len())?;
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

fn cases() -> Vec<[u32; 4]> {
    (0..256_u32)
        .map(|case| {
            core::array::from_fn::<u32, 4, _>(|k| case.wrapping_mul(0x9e37_79b9) >> (k * 7))
        })
        .collect()
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn by_value_array_iteration_matches_rust() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let cases = cases();
    for (&case, actual) in cases.iter().zip(run::<5>(&device, "iterated", &cases)?) {
        assert_eq!(actual, iterated(case), "{case:?}");
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn array_map_and_from_fn_match_rust() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let cases = cases();
    for (&case, actual) in cases.iter().zip(run::<6>(&device, "built", &cases)?) {
        assert_eq!(actual, built(case), "{case:?}");
    }
    Ok(())
}
