#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use metal_oxide::{Argument, Device, Error, GpuValue, LaunchConfig, Layout, Module, Pipeline};
use metal_oxide_artifact::Scalar;

#[repr(align(32))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Record {
    tag: u8,
    count: u32,
    pair: (f32, u16),
}

impl GpuValue for Record {
    const SIZE: usize = 16;
    const ALIGNMENT: usize = 4;
    fn layout() -> metal_oxide::Result<Layout> {
        Ok(Layout::record(
            "Record",
            vec![
                ("tag".into(), u8::layout()?),
                ("count".into(), u32::layout()?),
                ("pair".into(), <(f32, u16)>::layout()?),
            ],
        )?)
    }
    fn zeroed() -> Self {
        Self {
            tag: 0,
            count: 0,
            pair: (0.0, 0),
        }
    }
    fn encode(self, bytes: &mut [u8]) {
        bytes.fill(0);
        self.tag.encode(&mut bytes[0..1]);
        self.count.encode(&mut bytes[4..8]);
        self.pair.encode(&mut bytes[8..16]);
    }
    fn decode(bytes: &[u8]) -> Self {
        Self {
            tag: u8::decode(&bytes[0..1]),
            count: u32::decode(&bytes[4..8]),
            pair: <(f32, u16)>::decode(&bytes[8..16]),
        }
    }
}

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Pair { float value; ushort flags; };
struct Record { uchar tag; uint count; Pair pair; };
kernel void update(device const Record* input [[buffer(0)]], device Record* output [[buffer(1)]],
    constant Record& config [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) {
        Record value = input[i];
        value.count += config.count;
        value.pair.value += config.pair.value;
        output[i] = value;
    }
}
"#;

fn original() -> Record {
    Record {
        tag: 7,
        count: 9,
        pair: (2.0, 3),
    }
}
fn config() -> Record {
    Record {
        tag: 0,
        count: 5,
        pair: (3.0, 0),
    }
}
fn updated(times: u32) -> Record {
    Record {
        count: 9 + times * 5,
        pair: (2.0 + times as f32 * 3.0, 3),
        ..original()
    }
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn structured_storage_uses_gpu_stride_and_keeps_cpu_slices_valid() -> metal_oxide::Result<()> {
    assert_ne!(size_of::<Record>(), Record::SIZE);
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "update")?;
    for n in [0_u32, 1, 255, 256, 257] {
        let input = device.buffer_from_slice(&vec![original(); n as usize + 32])?;
        let mut out = device.buffer_from_slice(&vec![original(); input.len()])?;
        let view = input.as_slice();
        // SAFETY: separate initialized allocations cover n; each thread writes its own element.
        unsafe {
            device.launch(
                &pipeline,
                LaunchConfig::<32>::for_elements(n)?,
                &[
                    Argument::read(&input),
                    Argument::write(&mut out),
                    Argument::value(config())?,
                    Argument::value(n)?,
                ],
            )?;
        }
        assert_eq!(view, input.as_slice());
        assert!(
            out.as_slice()[..n as usize]
                .iter()
                .all(|value| *value == updated(1))
        );
        assert!(
            out.as_slice()[n as usize..]
                .iter()
                .all(|value| *value == original())
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn structured_chains_and_cancelled_submissions_keep_readback_current() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "update")?;
    let n = 257_u32;
    let launch = LaunchConfig::<32>::for_elements(n)?;
    let input = device.buffer_from_slice(&vec![original(); n as usize])?;
    let mut middle = device.buffer_zeroed::<Record>(n as usize)?;
    let mut out = device.buffer_zeroed::<Record>(n as usize)?;
    // SAFETY: the intermediate is read only after its write; allocations cover n unique writers.
    unsafe {
        device.submit(|batch| {
            batch.launch(
                &pipeline,
                launch,
                &[
                    Argument::read(&input),
                    Argument::write(&mut middle),
                    Argument::value(config())?,
                    Argument::value(n)?,
                ],
            )?;
            batch.launch(
                &pipeline,
                launch,
                &[
                    Argument::read(&middle),
                    Argument::write(&mut out),
                    Argument::value(config())?,
                    Argument::value(n)?,
                ],
            )
        })?
    }
    .wait()?;
    assert_eq!(out.as_slice(), vec![updated(2); n as usize]);
    for forget in [false, true] {
        out.as_mut_slice().fill(Record::zeroed());
        // SAFETY: input is read-only; output has n initialized elements and unique writers.
        let submission = unsafe {
            device.submit(|batch| {
                batch.launch(
                    &pipeline,
                    launch,
                    &[
                        Argument::read(&input),
                        Argument::write(&mut out),
                        Argument::value(config())?,
                        Argument::value(n)?,
                    ],
                )
            })?
        };
        if forget {
            std::mem::forget(submission);
        } else {
            drop(submission);
        }
        assert_eq!(out.as_slice(), vec![updated(1); n as usize]);
    }
    // SAFETY: CPU inspection precedes commit; captured output cannot escape to concurrent CPU access.
    unsafe {
        device.submit(|batch| {
            batch.launch(
                &pipeline,
                launch,
                &[
                    Argument::read(&middle),
                    Argument::write(&mut out),
                    Argument::value(config())?,
                    Argument::value(n)?,
                ],
            )?;
            assert_eq!(out.as_slice(), vec![updated(1); n as usize]);
            Ok(())
        })?
    }
    .wait()?;
    assert_eq!(out.as_slice(), vec![updated(2); n as usize]);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn structured_allocation_checks_and_discarded_batches_preserve_values() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    assert!(matches!(
        device.buffer_zeroed::<Record>(usize::MAX),
        Err(Error::LengthOverflow)
    ));
    let module = Module::from_source(&device, SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "update")?;
    let input = device.buffer_from_slice(&[original()])?;
    let mut out = device.buffer_from_slice(&[original()])?;
    // SAFETY: this valid command is discarded when the encoding closure returns its error.
    let result = unsafe {
        device.submit(|batch| {
            batch.launch(
                &pipeline,
                LaunchConfig::<32>::for_elements(1)?,
                &[
                    Argument::read(&input),
                    Argument::write(&mut out),
                    Argument::value(config())?,
                    Argument::value(1_u32)?,
                ],
            )?;
            Err(Error::Command("discard batch".into()))
        })
    };
    assert!(result.is_err());
    assert_eq!(out.as_slice(), [original()]);
    assert!(Argument::value(BadCodec).is_err());
    assert!(matches!(
        device.buffer_zeroed::<BadCodec>(1),
        Err(Error::Artifact(_))
    ));
    assert!(matches!(
        device.buffer_zeroed::<Huge>(0),
        Err(Error::BufferTooLarge { .. })
    ));
    Ok(())
}

#[derive(Clone, Copy)]
struct BadCodec;
impl GpuValue for BadCodec {
    const SIZE: usize = 16;
    const ALIGNMENT: usize = 4;
    fn layout() -> metal_oxide::Result<Layout> {
        Ok(Layout::scalar(Scalar::U32))
    }
    fn zeroed() -> Self {
        Self
    }
    fn encode(self, _: &mut [u8]) {
        panic!("invalid codec must not encode")
    }
    fn decode(_: &[u8]) -> Self {
        panic!("invalid codec must not decode")
    }
}

#[derive(Clone, Copy)]
struct Huge;
impl GpuValue for Huge {
    const SIZE: usize = 1 << 40;
    const ALIGNMENT: usize = 4;
    fn layout() -> metal_oxide::Result<Layout> {
        Ok(Layout::array(
            Layout::array(u32::layout()?, 1 << 28)?,
            1024,
        )?)
    }
    fn zeroed() -> Self {
        panic!("allocation limit must precede initialization")
    }
    fn encode(self, _: &mut [u8]) {
        unreachable!()
    }
    fn decode(_: &[u8]) -> Self {
        unreachable!()
    }
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn large_owned_constants_are_immutable_for_each_dispatch() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(
        &device,
        "#include <metal_stdlib>\nusing namespace metal;\nkernel void constants(device uint* out [[buffer(0)]], constant uint* values [[buffer(1)]]) { out[0] = values[0] + values[2047]; }",
    )?;
    let pipeline = Pipeline::new(&device, &module, "constants")?;
    let mut first = device.buffer_zeroed::<u32>(1)?;
    let mut second = device.buffer_zeroed::<u32>(1)?;
    let mut a = [0_u32; 2048];
    a[0] = 7;
    a[2047] = 11;
    let mut b = [0_u32; 2048];
    b[0] = 12;
    b[2047] = 3;
    // SAFETY: a single thread writes each initialized output; constants are immutable per dispatch.
    let submission = unsafe {
        device.submit(|batch| {
            batch.launch(
                &pipeline,
                LaunchConfig::<1>::for_elements(1)?,
                &[Argument::write(&mut first), Argument::value(a)?],
            )?;
            batch.launch(
                &pipeline,
                LaunchConfig::<1>::for_elements(1)?,
                &[Argument::write(&mut second), Argument::value(b)?],
            )
        })?
    };
    drop(submission);
    assert_eq!(first.as_slice(), [18]);
    assert_eq!(second.as_slice(), [15]);
    Ok(())
}
