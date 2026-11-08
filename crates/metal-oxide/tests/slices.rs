#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use metal_oxide::{
    Argument, Device, Launch, LaunchConfig, Module, Operation, Pipeline, ReadArgument, Result,
    WriteArgument, sequence,
};

const SOURCE: &str = "#include <metal_stdlib>
using namespace metal;
kernel void twice(const device uint* input [[buffer(0)]], device uint* output [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) output[i] = input[i] * 2u;
}
kernel void bytes(const device uchar* input [[buffer(0)]], device uchar* output [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) output[i] = input[i] + 1;
}
";

fn launch<I: ReadArgument<T>, O: WriteArgument<T>, T: metal_oxide::GpuValue>(
    pipeline: &Pipeline,
    input: I,
    output: O,
) -> Launch<(I, O, u32), 3> {
    let n = input.view().len() as u32;
    // SAFETY: the kernel guards n, the length of both views, and the views do not overlap.
    unsafe {
        Launch::new(
            Ok(pipeline),
            LaunchConfig::<64>::for_elements(n).unwrap(),
            (input, output, n),
            |(input, output, n)| {
                Ok([
                    Argument::read(input),
                    Argument::write(output),
                    Argument::value(*n)?,
                ])
            },
        )
    }
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn slices_bind_their_elements_at_an_offset() -> Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let twice = Pipeline::new(&device, &module, "twice")?;
    let input = device.buffer_from_slice(&(0..16_u32).collect::<Vec<_>>())?;
    let mut output = device.buffer_from_slice(&[u32::MAX; 16])?;
    launch(&twice, input.slice(2..10), output.slice_mut(4..12)).wait()?;
    let mut expected = [u32::MAX; 16];
    for (offset, value) in expected[4..12].iter_mut().enumerate() {
        *value = (offset as u32 + 2) * 2;
    }
    assert_eq!(output.read().wait(), &expected[..]);

    let bytes = Pipeline::new(&device, &module, "bytes")?;
    let input = device.buffer_from_slice(&(0..32_u8).collect::<Vec<_>>())?;
    let mut output = device.buffer_zeroed::<u8>(32)?;
    launch(&bytes, input.slice(1..20), output.slice_mut(3..22)).wait()?;
    let values = output.read().wait();
    assert_eq!(&values[..3], [0; 3]);
    assert_eq!(&values[3..22], (2..21).collect::<Vec<u8>>());
    assert_eq!(&values[22..], [0; 10]);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn views_of_one_buffer_follow_each_other_in_a_sequence() -> Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let twice = Pipeline::new(&device, &module, "twice")?;
    let input = device.buffer_from_slice(&(0..1024_u32).collect::<Vec<_>>())?;
    let mut middle = device.buffer_zeroed::<u32>(2048)?;
    let mut output = device.buffer_zeroed::<u32>(1024)?;
    sequence(|batch| {
        batch.push(launch(&twice, &input, middle.slice_mut(1024..)))?;
        batch.push(launch(&twice, middle.slice(1024..), &mut output))?;
        Ok(())
    })
    .wait()?;
    let expected = (0..1024_u32).map(|value| value * 4).collect::<Vec<_>>();
    assert_eq!(output.read().wait(), &expected[..]);
    assert!(middle.read().wait()[..1024].iter().all(|&value| value == 0));
    Ok(())
}
