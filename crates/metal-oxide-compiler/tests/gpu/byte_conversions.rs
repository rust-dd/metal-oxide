use super::*;

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/byte_conversions.rs";

type Bytes = ([u8; 8], [u8; 4], [u8; 2], [u16; 2], [u64; 5]);

fn byte_conversions((wide, signed, short): (u64, i32, i16)) -> Bytes {
    let little = wide.to_le_bytes();
    let big = signed.to_be_bytes();
    let native = short.to_ne_bytes();
    // SAFETY: integers and arrays of integers as wide have no invalid bit patterns.
    let halves = unsafe { core::mem::transmute::<u32, [u16; 2]>(signed as u32) };
    // SAFETY: as above.
    let packed = unsafe {
        core::mem::transmute::<[i16; 4], u64>([short, short.wrapping_neg(), short ^ 7, 1])
    };
    (
        little,
        big,
        native,
        halves,
        [
            u64::from_be_bytes(little),
            i32::from_le_bytes(big) as u64,
            i16::from_be_bytes(native) as u64,
            f32::from_le_bytes((wide as f32).to_be_bytes()).to_bits() as u64,
            packed,
        ],
    )
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn integer_byte_conversions_and_lane_transmutes_match_rust() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(&device, FIXTURE, "byte_conversions")?;
    let wide = [0, 1, 0x0123_4567_89ab_cdef, 1 << 63, u64::MAX, 0x8000_0000];
    let signed = [0, -1, 1, i32::MIN, i32::MAX, 0x1234_5678];
    let short = [0, -1, 1, i16::MIN, i16::MAX, 0x1234];
    let cases = (0..wide.len() * signed.len() * short.len())
        .map(|i| {
            (
                wide[i % wide.len()],
                signed[i / wide.len() % signed.len()],
                short[i / (wide.len() * signed.len())],
            )
        })
        .collect::<Vec<_>>();
    let input = device.buffer_from_slice(&cases)?;
    let mut output = device.buffer_zeroed::<Bytes>(cases.len())?;
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
    for (&case, actual) in cases.iter().zip(output.read().wait().iter()) {
        assert_eq!(*actual, byte_conversions(case), "{case:?}");
    }
    Ok(())
}
