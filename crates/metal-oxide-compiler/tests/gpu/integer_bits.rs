use super::*;

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/integer_bits.rs";
const PATTERNS: [u64; 14] = [
    0,
    1,
    2,
    3,
    0x7f,
    0x80,
    0xff,
    0x0123_4567_89ab_cdef,
    0x5555_5555_5555_5555,
    0x7fff_ffff_ffff_ffff,
    0x8000_0000_0000_0000,
    0xaaaa_aaaa_aaaa_aaaa,
    0xfedc_ba98_7654_3210,
    u64::MAX,
];
const SHIFTS: [u32; 14] = [0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 100, u32::MAX];

macro_rules! bits {
    ($test:ident, $kernel:literal, $ty:ty, $unsigned:ty) => {
        #[test]
        #[ignore = "requires a Metal device on Apple Silicon"]
        fn $test() -> metal_oxide::Result<()> {
            let device = Device::system_default()?;
            let pipeline = pipeline(&device, FIXTURE, $kernel)?;
            let patterns = PATTERNS.map(|p| p as $unsigned as $ty);
            let (mut inputs, mut shifts) = (Vec::new(), Vec::new());
            for (k, &a) in patterns.iter().enumerate() {
                for (j, &shift) in SHIFTS.iter().enumerate() {
                    inputs.extend([a, patterns[(k + j) % patterns.len()]]);
                    shifts.push(shift);
                }
            }
            let n = shifts.len();
            let input = device.buffer_from_slice(&inputs)?;
            let shift_buffer = device.buffer_from_slice(&shifts)?;
            let mut counts = device.buffer_zeroed::<u32>(3 * n)?;
            let mut values = device.buffer_zeroed::<$ty>(5 * n)?;
            // SAFETY: each active lane writes its own three counts and five values.
            unsafe {
                device.launch(
                    &pipeline,
                    LaunchConfig::<64>::for_elements(n as u32)?,
                    &[
                        Argument::read(&input),
                        Argument::read(&shift_buffer),
                        Argument::write(&mut counts),
                        Argument::write(&mut values),
                        Argument::value(n as u32)?,
                    ],
                )?;
            }
            for i in 0..n {
                let (a, b, shift) = (inputs[2 * i], inputs[2 * i + 1], shifts[i]);
                assert_eq!(
                    &counts.as_slice()[3 * i..3 * i + 3],
                    &[a.leading_zeros(), a.trailing_zeros(), a.count_ones()],
                    "{a:#x}"
                );
                assert_eq!(
                    &values.as_slice()[5 * i..5 * i + 5],
                    &[
                        a.reverse_bits(),
                        a.rotate_left(shift),
                        a.rotate_right(shift),
                        a.min(b),
                        a.max(b)
                    ],
                    "{a:#x} {b:#x} {shift}"
                );
            }
            Ok(())
        }
    };
}

bits!(u8_bit_operations_match_rust, "bits_u8", u8, u8);
bits!(u16_bit_operations_match_rust, "bits_u16", u16, u16);
bits!(u32_bit_operations_match_rust, "bits_u32", u32, u32);
bits!(u64_bit_operations_match_rust, "bits_u64", u64, u64);
bits!(i8_bit_operations_match_rust, "bits_i8", i8, u8);
bits!(i16_bit_operations_match_rust, "bits_i16", i16, u16);
bits!(i32_bit_operations_match_rust, "bits_i32", i32, u32);
