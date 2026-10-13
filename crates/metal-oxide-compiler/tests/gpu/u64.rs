use super::*;

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/u64.rs";

fn sm64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn splitmix64_matches_exact_cpu_streams_across_counter_boundaries() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(&device, FIXTURE, "splitmix")?;
    let n = 100_003_u32;
    for base in [0_u64, (1 << 32) - 5, (1 << 63) - 7, u64::MAX - 19] {
        for seed in [0_u32, 42, u32::MAX] {
            let mut output = device.buffer_from_slice(&vec![123_u64; n as usize + 32])?;
            // SAFETY: active lanes write distinct initialized u64 elements within n.
            unsafe {
                Launch::new(
                    Ok(&pipeline),
                    LaunchConfig::<128>::for_elements(n)?,
                    (&mut output, base, seed, n),
                    |(output, base, seed, n)| {
                        Ok([
                            Argument::write(output),
                            Argument::value(*base)?,
                            Argument::value(*seed)?,
                            Argument::value(*n)?,
                        ])
                    },
                )
            }
            .wait()?;
            let key = (seed as u64).wrapping_mul(0x9e3779b97f4a7c15);
            for (index, &actual) in output.read().wait()[..n as usize].iter().enumerate() {
                assert_eq!(
                    actual,
                    sm64(base.wrapping_add(index as u64) ^ key),
                    "base={base}, seed={seed}, index={index}"
                );
            }
            assert!(output.read().wait()[n as usize..].iter().all(|&x| x == 123));
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn u64_operations_and_saturating_casts_match_rust() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(&device, FIXTURE, "wide_ops")?;
    let mut output = device.buffer_zeroed::<u64>(15)?;
    for a in [0_u64, 1, 1 << 32, 1 << 63, u64::MAX - 1, u64::MAX] {
        for b in [0_u64, 1, 2, 1 << 32, u64::MAX] {
            for shift in [0_u32, 31, 32, 63, 64, 65, u32::MAX] {
                // SAFETY: a single lane owns all fifteen output elements.
                unsafe {
                    Launch::new(
                        Ok(&pipeline),
                        LaunchConfig::<1>::new(Dim3::x(1)),
                        (&mut output, a, b, shift, -123_i32),
                        |(output, a, b, shift, v4)| {
                            Ok([
                                Argument::write(output),
                                Argument::value(*a)?,
                                Argument::value(*b)?,
                                Argument::value(*shift)?,
                                Argument::value(*v4)?,
                            ])
                        },
                    )
                }
                .wait()?;
                let (sum, add) = a.overflowing_add(b);
                let (product, mul) = a.overflowing_mul(b);
                let (difference, sub) = a.overflowing_sub(b);
                assert_eq!(
                    output.read().wait(),
                    [
                        sum,
                        product,
                        difference,
                        (!a).wrapping_shl(shift),
                        a.wrapping_shr(shift),
                        -123_i32 as u64,
                        a as u32 as u64,
                        add as u64,
                        mul as u64,
                        sub as u64,
                        a.checked_div(b).unwrap_or(0),
                        a.checked_rem(b).unwrap_or(0),
                        a as f32 as u64,
                        a as i32 as u64,
                        match a {
                            u64::MAX => 7,
                            0x100000000 => 8,
                            _ => 9,
                        },
                    ],
                    "a={a}, b={b}, shift={shift}"
                );
            }
        }
    }
    for mode in [
        metal_oxide_artifact::MathMode::Precise,
        metal_oxide_artifact::MathMode::Fast,
    ] {
        let pipeline = pipeline_with_math_mode(&device, FIXTURE, "wide_casts", mode)?;
        let mut input = vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1.5,
        ];
        let limit = (u64::MAX as f32).to_bits();
        input.extend((limit - 4..=limit + 4).map(f32::from_bits));
        input.extend((0..4096_u32).map(|i| f32::from_bits(i.wrapping_mul(0x9e3779b9))));
        let n = input.len() as u32;
        let buffer = device.buffer_from_slice(&input)?;
        let mut output = device.buffer_zeroed::<u64>(input.len())?;
        // SAFETY: input/output have n initialized elements; each lane owns one output.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<128>::for_elements(n)?,
                (&buffer, &mut output, n),
                |(buffer, output, n)| {
                    Ok([
                        Argument::read(buffer),
                        Argument::write(output),
                        Argument::value(*n)?,
                    ])
                },
            )
        }
        .wait()?;
        for (&actual, &input) in output.read().wait().iter().zip(&input) {
            assert_eq!(
                actual,
                input as u64,
                "{mode:?}, input={:08x}",
                input.to_bits()
            );
        }
    }
    Ok(())
}
