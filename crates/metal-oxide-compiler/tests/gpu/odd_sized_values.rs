use super::*;

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/odd_sized_values.rs";

fn byte_triples(index: usize, pixel: [u8; 3]) -> [u32; 4] {
    let channel = pixel[index % 3];
    let dark = (0_u32..3).any(|c| pixel[(c + u32::from(channel)) as usize % 3] < 16);
    let wide = [
        u16::from(pixel[0]),
        u16::from(pixel[1]) * 3,
        u16::from(pixel[2]) * 5,
    ];
    let picked = (0_usize..2).fold(0_u16, |total, k| total ^ wide[(k + index) % 3]);
    let bytes = [pixel[2], pixel[1], pixel[0], channel, 7];
    [
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from(dark),
        u32::from(picked),
        u32::from(bytes[index % 5]),
    ]
}

fn pixels() -> Vec<[u8; 3]> {
    (0..256_u32)
        .map(|case| case.wrapping_mul(0x9e37_79b9).to_le_bytes())
        .map(|[a, b, c, _]| [a, b, c])
        .collect()
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn three_and_six_byte_values_match_rust() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pixels = pixels();
    let pipeline = pipeline(&device, FIXTURE, "byte_triples")?;
    let input = device.buffer_from_slice(&pixels)?;
    let mut output = device.buffer_zeroed::<[u32; 4]>(pixels.len())?;
    // SAFETY: the safe kernel bounds its writes and checks the input length.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<64>::for_elements(pixels.len() as u32)?,
            (&input, &mut output),
            |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
        )
    }
    .wait()?;
    for (index, (&pixel, actual)) in pixels.iter().zip(output.read().wait().iter()).enumerate() {
        assert_eq!(*actual, byte_triples(index, pixel), "{pixel:?}");
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn three_byte_buffer_elements_load_and_store() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pixels = pixels();
    let pipeline = pipeline(&device, FIXTURE, "reversed_triples")?;
    let input = device.buffer_from_slice(&pixels)?;
    let mut output = device.buffer_zeroed::<[u8; 3]>(pixels.len())?;
    let n = pixels.len() as u32;
    // SAFETY: thread i alone writes element i, and threads from n on write nothing.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<64>::for_elements(n)?,
            (&input, &mut output, n),
            |(input, output, n)| {
                Ok([
                    Argument::read(input),
                    Argument::write(output),
                    Argument::value(*n)?,
                ])
            },
        )
    }
    .wait()?;
    let expected = pixels
        .iter()
        .map(|&[a, b, c]| [c, b, a])
        .collect::<Vec<_>>();
    assert_eq!(output.read().wait().to_vec(), expected);
    Ok(())
}
