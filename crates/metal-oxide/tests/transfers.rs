#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::fmt::Debug;

use metal_oxide::{Device, Error, GpuValue, Operation, Result, sequence};

const LENGTHS: [usize; 10] = [0, 1, 3, 15, 16, 17, 255, 256, 257, 1_000_003];

fn copies<T: GpuValue + PartialEq + Debug>(device: &Device, value: fn(usize) -> T) -> Result<()> {
    for len in LENGTHS {
        let values = (0..len).map(value).collect::<Vec<_>>();
        let source = device.buffer_from_slice(&values)?;
        let mut destination = device.buffer_zeroed::<T>(len)?;
        destination.copy_from(&source).wait()?;
        assert_eq!(destination.read().wait(), &values[..], "{len} elements");
    }
    Ok(())
}

fn fills<T: GpuValue + PartialEq + Debug>(device: &Device, value: T) -> Result<()> {
    for len in LENGTHS {
        let mut buffer = device.buffer_zeroed::<T>(len)?;
        buffer.fill(value).wait()?;
        let values = buffer.read().wait();
        assert!(
            values.iter().all(|element| *element == value),
            "{len} of {value:?}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn copies_match_their_source_at_every_length_and_element_size() -> Result<()> {
    let device = Device::system_default()?;
    copies(&device, |i| i as u8)?;
    copies(&device, |i| i as f32 * 0.5)?;
    copies(&device, |i| [i as u8, (i >> 8) as u8, 7])?;
    copies(&device, |i| (i as u8, i as u64))?;
    let source = device.buffer_zeroed::<u32>(3)?;
    let mut shorter = device.buffer_zeroed::<u32>(2)?;
    let result = shorter.copy_from(&source).wait();
    assert!(matches!(result, Err(Error::InvalidLaunch(_))), "{result:?}");
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn fills_set_every_element_at_every_length_and_pattern_period() -> Result<()> {
    let device = Device::system_default()?;
    fills(&device, 7_u8)?;
    fills(&device, 1.5_f32)?;
    fills(&device, (5_u8, 9_u32))?;
    fills(&device, [1_u8, 2, 3])?;
    fills(&device, (1_u32, 2_u32, 3_u32))?;
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn copies_and_fills_follow_earlier_work_in_one_command_buffer() -> Result<()> {
    let device = Device::system_default()?;
    let values = (0..100_003_u32).collect::<Vec<_>>();
    let input = device.buffer_from_slice(&values)?;
    let mut output = device.buffer_zeroed::<u32>(values.len())?;
    let mut scratch = device.buffer_zeroed::<u32>(values.len())?;
    sequence(|batch| {
        batch.push(scratch.fill(9))?;
        batch.push(output.copy_from(&input))?;
        batch.push(scratch.copy_from(&output))?;
        Ok(())
    })
    .wait()?;
    assert_eq!(scratch.read().wait(), &values[..]);
    Ok(())
}
