#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

/// A host value with a kernel record's layout, encoded through the tuple of its fields.
macro_rules! record {
    ($name:ident($tuple:ty) { $($field:ident),* }) => {
        #[derive(Clone, Copy)]
        struct $name($tuple);

        impl metal_oxide::GpuValue for $name {
            const SIZE: usize = <$tuple as metal_oxide::GpuValue>::SIZE;
            const ALIGNMENT: usize = <$tuple as metal_oxide::GpuValue>::ALIGNMENT;

            fn layout() -> metal_oxide::Result<metal_oxide::Layout> {
                let metal_oxide::LayoutKind::Tuple { fields } =
                    <$tuple as metal_oxide::GpuValue>::layout()?.kind
                else {
                    unreachable!("record fields form a tuple")
                };
                let names = [$(stringify!($field)),*].map(str::to_owned);
                let fields = names.into_iter().zip(fields).map(|(name, field)| (name, field.layout));
                Ok(metal_oxide::Layout::record(stringify!($name), fields.collect())?)
            }

            fn zeroed() -> Self {
                Self(<$tuple as metal_oxide::GpuValue>::zeroed())
            }

            fn encode(self, bytes: &mut [u8]) {
                self.0.encode(bytes)
            }

            fn decode(bytes: &[u8]) -> Self {
                Self(<$tuple as metal_oxide::GpuValue>::decode(bytes))
            }
        }
    };
}

#[path = "gpu/aggregate_references.rs"]
mod aggregate_references;
#[path = "gpu/aggregates.rs"]
mod aggregates;
#[path = "gpu/artifact.rs"]
mod artifact;
#[path = "gpu/atomics.rs"]
mod atomics;
#[path = "gpu/checks.rs"]
mod checks;
#[path = "gpu/closures.rs"]
mod closures;
#[path = "gpu/collectives.rs"]
mod collectives;
#[path = "gpu/contracts.rs"]
mod contracts;
#[path = "gpu/control_flow.rs"]
mod control_flow;
#[path = "gpu/cooperation.rs"]
mod cooperation;
#[path = "gpu/cooperative.rs"]
mod cooperative;
#[path = "gpu/disjoint.rs"]
mod disjoint;
#[path = "gpu/dynamic_shared.rs"]
mod dynamic_shared;
#[path = "gpu/enums.rs"]
mod enums;
#[path = "gpu/fast_math.rs"]
mod fast_math;
#[path = "gpu/float.rs"]
mod float;
#[path = "gpu/generic_entries.rs"]
mod generic_entries;
#[path = "gpu/half.rs"]
mod half;
#[path = "gpu/integer_bits.rs"]
mod integer_bits;
#[path = "gpu/ir_aggregates.rs"]
mod ir_aggregates;
#[path = "gpu/iterators.rs"]
mod iterators;
#[path = "gpu/lengths.rs"]
mod lengths;
#[path = "gpu/lines.rs"]
mod lines;
#[path = "gpu/narrow.rs"]
mod narrow;
#[path = "gpu/native.rs"]
mod native;
#[path = "gpu/numerics.rs"]
mod numerics;
#[path = "gpu/proofs.rs"]
mod proofs;
#[path = "gpu/ranges.rs"]
mod ranges;
#[path = "gpu/read_write.rs"]
mod read_write;
#[path = "gpu/records.rs"]
mod records;
mod support;
#[path = "gpu/transcendentals.rs"]
mod transcendentals;
#[path = "gpu/u64.rs"]
mod u64;

use std::{
    future::{Future, IntoFuture},
    pin::Pin,
    task::{Context, Waker},
};

use metal_oxide::{
    Argument, Device, Dim3, DynamicLaunchConfig, Launch, LaunchConfig, Module, Operation, Pipeline,
    sequence,
};

fn started<O: IntoFuture<IntoFuture: Unpin>>(operation: O) -> O::IntoFuture {
    let mut future = operation.into_future();
    let _ = Pin::new(&mut future).poll(&mut Context::from_waker(Waker::noop()));
    future
}

fn pipeline(device: &Device, source: &str, entry: &str) -> metal_oxide::Result<Pipeline> {
    pipeline_with_math_mode(
        device,
        source,
        entry,
        metal_oxide_artifact::MathMode::Precise,
    )
}

fn pipeline_with_math_mode(
    device: &Device,
    source: &str,
    entry: &str,
    math_mode: metal_oxide_artifact::MathMode,
) -> metal_oxide::Result<Pipeline> {
    let (output, directory) = support::emit(
        source,
        &[
            "-C",
            "overflow-checks=off",
            "--metal-math-mode",
            math_mode.as_str(),
        ],
    );
    support::checked(output);
    let module = support::artifact::module(device, &directory)?;
    eprintln!("{entry}: {} ({})", device.name(), directory.display());
    Pipeline::new(device, &module, entry)
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn rust_vec_add_matches_cpu_and_preserves_padding() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/vec_add.rs",
        "vec_add",
    )?;
    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        verify_vec_add(&device, &pipeline, n, LaunchConfig::<32>::for_elements(n)?)?;
        verify_vec_add(&device, &pipeline, n, LaunchConfig::<256>::for_elements(n)?)?;
        verify_vec_add(
            &device,
            &pipeline,
            n,
            DynamicLaunchConfig::for_elements(n, pipeline.thread_execution_width() as u32)?,
        )?;
    }
    Ok(())
}

fn verify_vec_add(
    device: &Device,
    pipeline: &Pipeline,
    n: u32,
    config: impl Into<DynamicLaunchConfig> + Copy,
) -> metal_oxide::Result<()> {
    let block_size = config.into().block.x;
    let len = n as usize + block_size as usize;
    let a = (0..len)
        .map(|i| (i % 1024) as f32 * 0.25)
        .collect::<Vec<_>>();
    let b = (0..len)
        .map(|i| (i % 511) as f32 * -0.5)
        .collect::<Vec<_>>();
    let a_buffer = device.buffer_from_slice(&a)?;
    let b_buffer = device.buffer_from_slice(&b)?;
    let sentinel = -8192.0_f32;
    let mut output = device.buffer_from_slice(&vec![sentinel; len])?;
    // SAFETY: separate allocations cover n elements; each active thread writes its own index.
    unsafe {
        Launch::new(
            Ok(pipeline),
            config,
            (&a_buffer, &b_buffer, &mut output, n),
            |(a_buffer, b_buffer, output, n)| {
                Ok([
                    Argument::read(a_buffer),
                    Argument::read(b_buffer),
                    Argument::write(output),
                    Argument::value::<u32>(*n)?,
                ])
            },
        )
    }
    .wait()?;
    for (i, actual) in output.read().wait()[..n as usize].iter().enumerate() {
        assert_eq!(*actual, a[i] + b[i], "n={n}, block={block_size}, i={i}");
    }
    assert!(
        output.read().wait()[n as usize..]
            .iter()
            .all(|v| *v == sentinel)
    );
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn rust_nested_branches_loop_and_early_return_match_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/m2_control_flow.rs",
        "control_flow",
    )?;
    let n = 257_u32;
    let sentinel = u32::MAX;
    let mut output = device.buffer_from_slice(&vec![sentinel; n as usize + 32])?;
    // SAFETY: the kernel guards n and each active thread writes a separate output element.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<32>::for_elements(n)?,
            (&mut output, n),
            |(output, n)| Ok([Argument::write(output), Argument::value::<u32>(*n)?]),
        )
    }
    .wait()?;
    for (i, &actual) in output.read().wait()[..n as usize].iter().enumerate() {
        let sum = (i as u32 / 2) * 10 + (i as u32 % 2) * 3;
        let expected = if i == 0 {
            99
        } else if sum > 30 {
            sum - 1
        } else if sum > 10 {
            sum + 2
        } else {
            sum
        };
        assert_eq!(actual, expected, "i={i}");
    }
    assert!(
        output.read().wait()[n as usize..]
            .iter()
            .all(|v| *v == sentinel)
    );
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn rust_concrete_trait_and_const_generic_helpers_execute() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/generics.rs",
        "helpers",
    )?;
    let mut float = device.buffer_from_slice(&[-1.0_f32; 2])?;
    let mut integer = device.buffer_zeroed::<u32>(1)?;
    // SAFETY: one thread writes float[grid_dim.x = 1] and integer[0] in separate allocations.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<1>::new(Dim3::x(1)),
            (&mut float, &mut integer),
            |(float, integer)| Ok([Argument::write(float), Argument::write(integer)]),
        )
    }
    .wait()?;
    assert_eq!(float.read().wait(), [-1.0, 3.0]);
    assert_eq!(integer.read().wait(), [6]);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn rust_signed_integer_math_wraps_and_masks_shifts() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/m2_integer.rs",
        "integer_math",
    )?;
    for (a, b, shift) in [
        (i32::MAX, 1, 0),
        (i32::MIN, -1, 31),
        (-123_456_789, 73, 32),
        (5, -7, 33),
        (i32::MIN, i32::MIN, 63),
    ] {
        let mut output = device.buffer_zeroed::<i32>(6)?;
        // SAFETY: one thread writes six output elements; scalar arguments match the generated slots.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<1>::new(Dim3::x(1)),
                (&mut output, a, b, shift),
                |(output, a, b, shift)| {
                    Ok([
                        Argument::write(output),
                        Argument::value::<i32>(*a)?,
                        Argument::value::<i32>(*b)?,
                        Argument::value::<u32>(*shift)?,
                    ])
                },
            )
        }
        .wait()?;
        assert_eq!(
            output.read().wait(),
            [
                a.wrapping_add(b),
                a.wrapping_sub(b),
                a.wrapping_mul(b),
                a.wrapping_neg(),
                a.wrapping_shl(shift),
                a.wrapping_shr(shift)
            ]
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn rust_float_operations_are_not_contracted_into_fma() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/m2_float.rs",
        "float_math",
    )?;
    let a = 1.0_f32 + f32::EPSILON;
    let b = 1.0_f32 - f32::EPSILON;
    let c = -1.0_f32;
    assert_ne!(a.mul_add(b, c), (a * b) + c);
    let mut output = device.buffer_zeroed::<f32>(1)?;
    // SAFETY: one thread writes one element and receives three f32 scalar bindings.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<1>::new(Dim3::x(1)),
            (&mut output, a, b, c),
            |(output, a, b, c)| {
                Ok([
                    Argument::write(output),
                    Argument::value::<f32>(*a)?,
                    Argument::value::<f32>(*b)?,
                    Argument::value::<f32>(*c)?,
                ])
            },
        )
    }
    .wait()?;
    assert_eq!(output.read().wait()[0].to_bits(), ((a * b) + c).to_bits());
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn rust_cuda_coordinates_match_cpu_in_1d_2d_and_3d() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/m2_geometry.rs",
        "geometry",
    )?;
    for config in [
        LaunchConfig::<5>::new(Dim3::x(3)).into(),
        LaunchConfig::<3, 2>::new(Dim3::xy(2, 3)).into(),
        LaunchConfig::<2, 3, 2>::new(Dim3::new(2, 2, 3)).into(),
        DynamicLaunchConfig::new(
            Dim3::xy(2, 3),
            Dim3::x(pipeline.thread_execution_width() as u32),
        ),
    ] {
        let mut output = device.buffer_zeroed::<u32>(config.total_threads()? as usize * 15)?;
        // SAFETY: each global thread owns one 15-element record in the fully sized buffer.
        unsafe {
            Launch::new(Ok(&pipeline), config, (&mut output,), |(output,)| {
                Ok([Argument::write(output)])
            })
        }
        .wait()?;
        let width = config.grid.x * config.block.x;
        let height = config.grid.y * config.block.y;
        for (i, actual) in output.read().wait().as_chunks::<15>().0.iter().enumerate() {
            let x = i as u32 % width;
            let y = i as u32 / width % height;
            let z = i as u32 / (width * height);
            assert_eq!(
                *actual,
                [
                    x,
                    y,
                    z,
                    x % config.block.x,
                    y % config.block.y,
                    z % config.block.z,
                    x / config.block.x,
                    y / config.block.y,
                    z / config.block.z,
                    config.block.x,
                    config.block.y,
                    config.block.z,
                    config.grid.x,
                    config.grid.y,
                    config.grid.z
                ]
            );
        }
    }
    Ok(())
}
