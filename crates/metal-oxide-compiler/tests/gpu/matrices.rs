use super::*;
use metal_oxide::{BF16, Error, F16, GpuValue};

const MATRICES: &str = "crates/metal-oxide-compiler/tests/fixtures/matrices.rs";
const UNIFORMITY: &str = "crates/metal-oxide-compiler/tests/fixtures/matrix_uniformity.rs";
const FEATURES: &[&str] = &["simdgroup_matrix", "bfloat16"];

trait Element: GpuValue + Copy + PartialEq + std::fmt::Debug {
    fn from_f32(value: f32) -> Self;
}

impl Element for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }
}

impl Element for F16 {
    fn from_f32(value: f32) -> Self {
        F16::from_f32(value)
    }
}

impl Element for BF16 {
    fn from_f32(value: f32) -> Self {
        BF16::from_f32(value)
    }
}

/// The index whose value the transpose of `index`'s 8x8 block puts at `index`.
fn transposed(index: usize) -> usize {
    let (row, column) = (index / 8, index % 8);
    (row / 8 * 8 + column) * 8 + row % 8
}

/// The line of the uniformity fixture that contains `text`, as a check location reports it.
fn line(text: &str) -> u32 {
    let source = include_str!("../fixtures/matrix_uniformity.rs");
    source.lines().position(|line| line.contains(text)).unwrap() as u32 + 1
}

/// Multiples of 1/4 and of 1/2: their products and sums stay exact in every element type,
/// whatever order Metal accumulates in.
fn patterns(left: usize, right: usize) -> (Vec<f32>, Vec<f32>) {
    let quarters = (0..left).map(|i| ((i % 7) as f32 - 3.0) * 0.25);
    let halves = (0..right).map(|i| ((i % 5) as f32 - 2.0) * 0.5);
    (quarters.collect(), halves.collect())
}

fn round_trip<T: Element>(device: &Device, entry: &str) -> metal_oxide::Result<()> {
    let Some(pipeline) = supported_pipeline(device, MATRICES, entry, FEATURES)? else {
        return Ok(());
    };
    // Paths: device copy, load^T, store^T; shared copy; scratch store^T, load^T; shared store^T.
    let paths = [false, true, true, false, true, true, true];
    let input = (0..128).map(|i| T::from_f32(i as f32)).collect::<Vec<_>>();
    let source = device.buffer_from_slice(&input)?;
    let mut output = device.buffer_zeroed::<T>(paths.len() * 128)?;
    // SAFETY: one block reads 128 elements, writes 128 per path, stages 256 per threadgroup buffer.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<64>::new(Dim3::x(1)),
            (&source, &mut output),
            |(source, output)| {
                Ok([
                    Argument::read(source),
                    Argument::write(output),
                    Argument::threadgroup::<T>(256)?,
                ])
            },
        )
        .wait()?;
    }
    let output = output.read().wait();
    for index in 0..128 {
        for (path, &flipped) in paths.iter().enumerate() {
            let expected = input[if flipped { transposed(index) } else { index }];
            assert_eq!(
                output[path * 128 + index],
                expected,
                "{entry}: path {path}, element {index}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn matrix_loads_and_stores_round_trip_through_device_and_threadgroup_memory()
-> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    round_trip::<f32>(&device, "round_trip_f32")?;
    round_trip::<F16>(&device, "round_trip_f16")?;
    round_trip::<BF16>(&device, "round_trip_bf16")
}

fn product<A: Element, B: Element, C: Element>(
    device: &Device,
    entry: &str,
) -> metal_oxide::Result<()> {
    let Some(pipeline) = supported_pipeline(device, MATRICES, entry, FEATURES)? else {
        return Ok(());
    };
    let (a, b) = patterns(64, 64);
    let c = (0..64).map(|i| (i % 3) as f32).collect::<Vec<_>>();
    let expected = (0..64)
        .map(|index| {
            let (row, column) = (index / 8, index % 8);
            let sum = (0..8)
                .map(|k| a[row * 8 + k] * b[k * 8 + column])
                .sum::<f32>();
            C::from_f32(sum + c[index])
        })
        .collect::<Vec<_>>();
    let a = device.buffer_from_slice(&a.into_iter().map(A::from_f32).collect::<Vec<_>>())?;
    let b = device.buffer_from_slice(&b.into_iter().map(B::from_f32).collect::<Vec<_>>())?;
    let c = device.buffer_from_slice(&c.into_iter().map(C::from_f32).collect::<Vec<_>>())?;
    let mut d = device.buffer_zeroed::<C>(64)?;
    // SAFETY: each buffer holds one 8x8 matrix; one SIMD group runs.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<32>::new(Dim3::x(1)),
            (&a, &b, &c, &mut d),
            |(a, b, c, d)| {
                Ok([
                    Argument::read(a),
                    Argument::read(b),
                    Argument::read(c),
                    Argument::write(d),
                ])
            },
        )
        .wait()?;
    }
    assert_eq!(d.read().wait()[..], expected[..], "{entry}");
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn multiply_accumulate_matches_cpu_products_in_every_precision() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    product::<f32, f32, f32>(&device, "multiply_f32")?;
    product::<F16, F16, f32>(&device, "multiply_f16")?;
    product::<BF16, BF16, f32>(&device, "multiply_bf16")?;
    product::<F16, F16, F16>(&device, "multiply_f16_accumulator")?;
    product::<BF16, BF16, BF16>(&device, "multiply_bf16_accumulator")?;
    product::<F16, BF16, f32>(&device, "multiply_f16_bf16")?;
    product::<f32, F16, f32>(&device, "multiply_f32_f16")
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn a_read_write_buffer_is_multiplied_and_transposed_in_place() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let Some(pipeline) = supported_pipeline(&device, MATRICES, "in_place_f32", FEATURES)? else {
        return Ok(());
    };
    let (c, b) = patterns(64, 64);
    let expected = (0..64)
        .map(|index| {
            // The result at (row, column) is element (column, row) of C * B + C.
            let (row, column) = (index / 8, index % 8);
            let products = (0..8).map(|k| c[column * 8 + k] * b[k * 8 + row]);
            products.sum::<f32>() + c[column * 8 + row]
        })
        .collect::<Vec<_>>();
    let b = device.buffer_from_slice(&b)?;
    let mut c = device.buffer_from_slice(&c)?;
    // SAFETY: each buffer holds one 8x8 matrix; one SIMD group runs.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<32>::new(Dim3::x(1)),
            (&b, &mut c),
            |(b, c)| Ok([Argument::read(b), Argument::read_write(c)]),
        )
        .wait()?;
    }
    assert_eq!(c.read().wait()[..], expected[..]);
    Ok(())
}

fn gemm(device: &Device, entry: &str, nt: bool) -> metal_oxide::Result<()> {
    let Some(pipeline) = supported_pipeline(device, MATRICES, entry, FEATURES)? else {
        return Ok(());
    };
    for (m, n, k) in [
        (37_usize, 29_usize, 19_usize),
        (32, 32, 16),
        (8, 8, 8),
        (1, 1, 1),
    ] {
        let (a, b) = patterns(m * k, k * n);
        let expected = (0..m * n)
            .map(|index| {
                let (row, column) = (index / n, index % n);
                (0..k)
                    .map(|i| {
                        let right = if nt {
                            b[column * k + i]
                        } else {
                            b[i * n + column]
                        };
                        a[row * k + i] * right
                    })
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let a = device.buffer_from_slice(&a.into_iter().map(F16::from_f32).collect::<Vec<_>>())?;
        let b = device.buffer_from_slice(&b.into_iter().map(F16::from_f32).collect::<Vec<_>>())?;
        let mut c = device.buffer_zeroed::<f32>(m * n)?;
        let shape = (m as u32, n as u32, k as u32);
        // SAFETY: the buffers hold m*k, k*n and m*n elements; the grid covers C in 16x16 tiles.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<64>::new(Dim3::xy(n.div_ceil(16) as u32, m.div_ceil(16) as u32)),
                (&a, &b, &mut c, shape),
                |(a, b, c, (m, n, k))| {
                    Ok([
                        Argument::read(a),
                        Argument::read(b),
                        Argument::write(c),
                        Argument::value(*m)?,
                        Argument::value(*n)?,
                        Argument::value(*k)?,
                    ])
                },
            )
            .wait()?;
        }
        assert_eq!(c.read().wait()[..], expected[..], "{entry} {m}x{n}x{k}");
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn tiled_gemm_matches_cpu_for_ragged_and_transposed_inputs() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    gemm(&device, "gemm_f16", false)?;
    gemm(&device, "gemm_nt_f16", true)
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn group_uniform_control_and_checks_guard_matrix_operations() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let features = &["simdgroup_matrix"];
    let Some(branch) = supported_pipeline(&device, UNIFORMITY, "group_branch", features)? else {
        return Ok(());
    };
    let mut output = device.buffer_zeroed::<f32>(128)?;
    // SAFETY: each of the two SIMD groups stores its own 8x8 block.
    unsafe {
        Launch::new(
            Ok(&branch),
            LaunchConfig::<64>::new(Dim3::x(1)),
            (&mut output,),
            |(output,)| Ok([Argument::write(output)]),
        )
        .wait()?;
    }
    let expected = (0..128)
        .map(|i| if i < 64 { 1.0 } else { 2.0 })
        .collect::<Vec<f32>>();
    assert_eq!(output.read().wait()[..], expected[..]);
    let checked = supported_pipeline(&device, UNIFORMITY, "checked_group", features)?.unwrap();
    for groups in [2_u32, 1] {
        let mut output = device.buffer_zeroed::<f32>(128)?;
        // SAFETY: each SIMD group stores its own 8x8 block.
        let result = unsafe {
            Launch::new(
                Ok(&checked),
                LaunchConfig::<64>::new(Dim3::x(1)),
                (&mut output, groups),
                |(output, groups)| Ok([Argument::write(output), Argument::value(*groups)?]),
            )
            .wait()
            .map(drop)
        };
        let stored = output.read().wait();
        if groups == 2 {
            result?;
            assert!(stored.iter().all(|&x| x == 1.0));
            continue;
        }
        let Err(Error::KernelFailed(failure)) = result else {
            panic!("expected a kernel failure, got {result:?}");
        };
        assert_eq!(failure.kernel, "checked_group");
        assert_eq!(failure.message, "SIMD group outside the launch contract");
        assert!(
            failure.file.ends_with("fixtures/matrix_uniformity.rs"),
            "{failure}"
        );
        assert_eq!(failure.line, line("assert!(group < groups"));
        // The group that passes its check still stores its block; the other one does not.
        assert!(stored[..64].iter().all(|&x| x == 1.0), "{stored:?}");
        assert!(stored[64..].iter().all(|&x| x == 0.0), "{stored:?}");
    }
    Ok(())
}
