use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn primitive_cpu_views_use_the_metal_allocation() -> Result<()> {
    let device = Device::system_default()?;
    fn verify<T: GpuValue>(device: &Device) -> Result<()> {
        for len in [0, 1, 257] {
            let mut buffer = device.buffer_zeroed::<T>(len)?;
            let address = buffer.storage.contents().cast::<T>().as_ptr();
            assert_eq!(buffer.as_slice().as_ptr(), address);
            assert_eq!(buffer.as_mut_slice().as_mut_ptr(), address);
            assert_eq!(buffer.as_slice().len(), len);
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
    let address = buffer.as_slice().as_ptr();
    let start = std::time::Instant::now();
    for _ in 0..100 {
        buffer.upload();
        std::hint::black_box(buffer.as_slice());
    }
    eprintln!(
        "shared f32 buffer: create={create:?}, 100 upload/read views={:?}",
        start.elapsed()
    );
    assert!(
        buffer
            .as_slice()
            .iter()
            .zip(&data)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    buffer.as_mut_slice()[0] = -0.0;
    buffer.upload();
    assert_eq!(buffer.as_slice().as_ptr(), address);
    // SAFETY: no GPU work is pending and f32 bits are readable as u32.
    let bits = unsafe { buffer.storage.contents().cast::<u32>().as_ptr().read() };
    assert_eq!(bits, 0x8000_0000);
    Ok(())
}
