use super::*;

#[test]
fn native_layout_requires_a_builtin_type_identity() {
    assert!(native_layout::<crate::BF16>());
    assert!(!native_layout::<[crate::BF16; 1]>());
    assert!(!native_layout::<(crate::BF16,)>());
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn primitive_cpu_views_use_the_metal_allocation() -> Result<()> {
    let device = Device::system_default()?;
    fn verify<T: GpuValue>(device: &Device) -> Result<()> {
        for len in [0, 1, 257] {
            let mut buffer = device.buffer_zeroed::<T>(len)?;
            let address = buffer.storage.contents().cast::<T>().as_ptr();
            assert_eq!(buffer.read().wait().as_ptr(), address);
            assert_eq!(buffer.write().wait().as_mut_ptr(), address);
            assert_eq!(buffer.read().wait().len(), len);
            assert!(address.is_aligned());
            // SAFETY: the initialized allocation is idle and contains len elements.
            let bytes = unsafe { std::slice::from_raw_parts(address.cast::<u8>(), len * T::SIZE) };
            assert!(bytes.iter().all(|byte| *byte == 0));
        }
        Ok(())
    }
    verify::<u8>(&device)?;
    verify::<i8>(&device)?;
    verify::<u16>(&device)?;
    verify::<i16>(&device)?;
    verify::<u32>(&device)?;
    verify::<u64>(&device)?;
    verify::<i32>(&device)?;
    verify::<f32>(&device)?;
    verify::<crate::F16>(&device)?;
    verify::<crate::BF16>(&device)?;
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn bfloat_shared_views_preserve_bits_without_mirror_copies() -> Result<()> {
    use std::{
        hint::black_box,
        time::{Duration, Instant},
    };

    let device = Device::system_default()?;
    let values = (0..1_000_003)
        .map(|i| crate::BF16::from_bits(i as u16))
        .collect::<Vec<_>>();
    let mut direct = device.buffer_from_slice(&values)?;
    let mut mirrored = Buffer {
        mirror: Some(Mirror::new(values.clone(), Location::Cpu)),
        ..device.buffer_zeroed::<crate::BF16>(values.len())?
    };
    let address = direct.storage.contents().cast::<crate::BF16>().as_ptr();
    assert!(direct.mirror.is_none());
    assert_eq!(direct.read().wait().as_ptr(), address);

    let sample = |buffer: &mut Buffer<crate::BF16>| {
        buffer.write().wait().copy_from_slice(&values);
        let start = Instant::now();
        buffer.upload();
        let upload = start.elapsed();
        // Invalidate the reference mirror to measure readback of the same idle native bytes.
        buffer.prepare(Access::Write);
        let start = Instant::now();
        black_box(buffer.read().wait());
        (upload, start.elapsed())
    };
    let mut direct_times = Vec::new();
    let mut mirror_times = Vec::new();
    for round in 0..25 {
        let samples = if round % 2 == 0 {
            (sample(&mut direct), sample(&mut mirrored))
        } else {
            let mirrored = sample(&mut mirrored);
            (sample(&mut direct), mirrored)
        };
        if round >= 5 {
            direct_times.push(samples.0);
            mirror_times.push(samples.1);
        }
    }
    let median = |samples: &[(Duration, Duration)], readback: bool| {
        let mut times = samples
            .iter()
            .map(|&(upload, read)| if readback { read } else { upload })
            .collect::<Vec<_>>();
        times.sort_unstable();
        times[times.len() / 2]
    };
    eprintln!(
        "BF16 {} elements, 20 samples: direct upload={:?} read={:?}; mirror upload={:?} read={:?}",
        values.len(),
        median(&direct_times, false),
        median(&direct_times, true),
        median(&mirror_times, false),
        median(&mirror_times, true),
    );
    for buffer in [&direct, &mirrored] {
        assert!(
            buffer
                .read()
                .wait()
                .iter()
                .zip(&values)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );
    }
    direct.write().wait()[0] = crate::BF16::from_bits(0x8000);
    direct.upload();
    assert_eq!(direct.read().wait().as_ptr(), address);
    // SAFETY: this idle initialized allocation stores BF16 transparently as u16.
    assert_eq!(unsafe { address.cast::<u16>().read() }, 0x8000);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn shared_primitive_views_preserve_bits_and_reuse_storage() -> Result<()> {
    let device = Device::system_default()?;
    let data = (0..1_000_003)
        .map(|i| f32::from_bits(i * 4093))
        .collect::<Vec<_>>();
    let start = std::time::Instant::now();
    let mut buffer = device.buffer_from_slice(&data)?;
    let create = start.elapsed();
    let address = buffer.read().wait().as_ptr();
    let start = std::time::Instant::now();
    for _ in 0..100 {
        buffer.upload();
        std::hint::black_box(buffer.read().wait());
    }
    eprintln!(
        "shared f32 buffer: create={create:?}, 100 upload/read views={:?}",
        start.elapsed()
    );
    assert!(
        buffer
            .read()
            .wait()
            .iter()
            .zip(&data)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    buffer.write().wait()[0] = -0.0;
    buffer.upload();
    assert_eq!(buffer.read().wait().as_ptr(), address);
    // SAFETY: no GPU work is pending and f32 bits are readable as u32.
    let bits = unsafe { buffer.storage.contents().cast::<u32>().as_ptr().read() };
    assert_eq!(bits, 0x8000_0000);
    Ok(())
}
