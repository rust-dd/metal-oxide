use super::*;
use metal_oxide::{BF16, F16, GpuValue};

trait Half: GpuValue + std::fmt::Debug {
    fn from_bits(value: u16) -> Self;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn bits(self) -> u16;
    fn is_subnormal(self) -> bool;
}

macro_rules! half {
    ($ty:ty) => {
        impl Half for $ty {
            fn from_bits(value: u16) -> Self {
                Self::from_bits(value)
            }
            fn from_f64(value: f64) -> Self {
                Self::from_f64(value)
            }
            fn to_f64(self) -> f64 {
                self.to_f64()
            }
            fn bits(self) -> u16 {
                self.to_bits()
            }
            fn is_subnormal(self) -> bool {
                self.classify() == core::num::FpCategory::Subnormal
            }
        }
    };
}
half!(F16);
half!(BF16);

fn arithmetic<T: Half>(device: &Device, module: &Module, name: &str) -> metal_oxide::Result<()> {
    let pipeline = Pipeline::new(device, module, name)?;
    let values = (0..4096_u32)
        .map(|i| {
            [
                T::from_bits(i.wrapping_mul(0x9e37) as u16),
                T::from_bits(i.wrapping_mul(0x79b9).wrapping_add(17) as u16),
            ]
        })
        .chain(
            [
                [0.0, -0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [f64::INFINITY, f64::INFINITY],
                [f64::NAN, 1.0],
            ]
            .map(|pair| pair.map(T::from_f64)),
        )
        .collect::<Vec<_>>();
    let input = device.buffer_from_slice(&values)?;
    let mut output = device.buffer_zeroed::<[T; 8]>(values.len())?;
    // SAFETY: the safe kernel bounds its writes and checks the input length.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<64>::for_elements(values.len() as u32)?,
            (&input, &mut output),
            |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
        )
    }
    .wait()?;
    let output = output.read().wait();
    for (pair, row) in values.into_iter().zip(output.iter()) {
        let [a, b] = pair.map(T::to_f64);
        let calculate = |a: f64, b: f64| {
            [
                a + b,
                a - b,
                a * b,
                a / b,
                a % b,
                -a,
                f64::from(a < b),
                f64::from(a == b),
            ]
        };
        let flush = |value: T| {
            if value.is_subnormal() {
                0.0_f64.copysign(value.to_f64())
            } else {
                value.to_f64()
            }
        };
        let exact = calculate(a, b);
        let flushed = calculate(flush(pair[0]), flush(pair[1]));
        for (operation, &actual) in row.iter().enumerate() {
            let matches = [exact[operation], flushed[operation]]
                .into_iter()
                .any(|value| {
                    let expected = T::from_f64(value);
                    actual.bits() == expected.bits()
                        || (actual.to_f64().is_nan() && expected.to_f64().is_nan())
                        || (expected.is_subnormal() && actual.to_f64() == 0.0)
                });
            assert!(
                matches,
                "{name} {a} {b} op={operation}: {actual:?}, expected {:?}",
                T::from_f64(exact[operation])
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn half_and_bfloat_arithmetic_match_rounded_scalar_operations() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/half_arithmetic.rs",
        &[],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )?;
    assert_eq!(abi.required_features, ["bfloat16"]);
    let module = support::artifact::module(&device, &directory)?;
    arithmetic::<F16>(&device, &module, "f16_arithmetic")?;
    arithmetic::<F16>(&device, &module, "primitive")?;
    arithmetic::<BF16>(&device, &module, "bf16_arithmetic")?;
    let values = [
        0.0,
        -0.0,
        -1.5,
        2.5,
        65504.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ]
    .map(F16::from_f64);
    let input = device.buffer_from_slice(&values)?;
    let mut output = device.buffer_zeroed::<(u64, i32, F16)>(values.len())?;
    let pipeline = Pipeline::new(&device, &module, "half_casts")?;
    // SAFETY: the safe kernel bounds its writes and checks the input length.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<32>::for_elements(values.len() as u32)?,
            (&input, &mut output),
            |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
        )
    }
    .wait()?;
    for (value, &(unsigned, signed, half)) in values.into_iter().zip(output.read().wait().iter()) {
        let value = value.to_f64();
        assert_eq!(unsigned, value as u64);
        assert_eq!(signed, value as i32);
        assert_eq!(
            half.to_bits(),
            F16::from_f64((value as u64) as f64).to_bits()
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn bfloat_storage_widens_every_pattern_and_rounds_f32_midpoints() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/half_arithmetic.rs",
        &[],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;
    let values = (0..=u16::MAX).map(BF16::from_bits).collect::<Vec<_>>();
    let input = device.buffer_from_slice(&values)?;
    let mut output = device.buffer_zeroed::<f32>(values.len())?;
    let pipeline = Pipeline::new(&device, &module, "bfloat_widen")?;
    // SAFETY: the safe kernel bounds its writes and checks the input length.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<256>::for_elements(values.len() as u32)?,
            (&input, &mut output),
            |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
        )
    }
    .wait()?;
    for (expected, &actual) in values.into_iter().zip(output.read().wait().iter()) {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_f32().to_bits(),
                "bf16 {:04x}",
                expected.to_bits()
            );
        }
    }
    let values = (0..=u16::MAX)
        .flat_map(|bits| {
            [0x7fff, 0x8000, 0x8001].map(move |tail| f32::from_bits((u32::from(bits) << 16) | tail))
        })
        .collect::<Vec<_>>();
    let input = device.buffer_from_slice(&values)?;
    let mut output = device.buffer_zeroed::<BF16>(values.len())?;
    let pipeline = Pipeline::new(&device, &module, "bfloat_round")?;
    // SAFETY: the safe kernel bounds its writes and checks the input length.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<256>::for_elements(values.len() as u32)?,
            (&input, &mut output),
            |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
        )
    }
    .wait()?;
    for (value, &actual) in values.into_iter().zip(output.read().wait().iter()) {
        let expected = BF16::from_f32(value);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "f32 {:08x}",
                value.to_bits()
            );
        }
    }
    Ok(())
}
