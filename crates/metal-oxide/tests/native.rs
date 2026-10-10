#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use metal_oxide::{Argument, Buffer, Device, Error, LaunchConfig, Module, Pipeline};
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
};

const SOURCE: &str = "#include <metal_stdlib>\nusing namespace metal;\nkernel void add(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) { if (i < n) out[i] += 7; }";

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn native_handoffs_keep_storage_and_wait_for_cancelled_work() -> metal_oxide::Result<()> {
    let raw_device = MTLCreateSystemDefaultDevice().unwrap();
    let device = Device::from_native(raw_device.clone())?;
    assert_eq!(device.as_native().registryID(), raw_device.registryID());
    let module = Module::from_source(&device, SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "add")?;
    let n = 100_003;
    for completion in 0..3 {
        let raw = raw_device
            .newBufferWithLength_options((n + 32) * 4, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let address = raw.contents().as_ptr();
        let queue = raw_device.newCommandQueue().unwrap();
        let command = queue.commandBuffer().unwrap();
        let encoder = command.blitCommandEncoder().unwrap();
        encoder.fillBuffer_range_value(&raw, NSRange::new(0, raw.length()), 1);
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        // SAFETY: native initialization completed; the retained alias is only dropped,
        // and no external access occurs until into_native returns ownership.
        let mut values = unsafe { Buffer::<u32>::from_native(&device, raw.clone(), n + 32)? };
        drop(raw);
        assert_eq!(values.as_slice().as_ptr().cast_mut().cast(), address);
        // SAFETY: each active lane updates its own element and the kernel guards padding.
        let submission = unsafe {
            device.submit(|batch| {
                batch.launch(
                    &pipeline,
                    LaunchConfig::<32>::for_elements(n as u32)?,
                    &[
                        Argument::read_write(&mut values),
                        Argument::value(n as u32)?,
                    ],
                )
            })?
        };
        match completion {
            0 => {
                submission.wait()?;
            }
            1 => drop(submission),
            _ => std::mem::forget(submission),
        }
        let raw = values.into_native();
        assert_eq!(raw.contents().as_ptr(), address);
        // SAFETY: owning export waited for every access; all words were initialized.
        let actual =
            unsafe { std::slice::from_raw_parts(raw.contents().cast::<u32>().as_ptr(), n + 32) };
        assert!(actual[..n].iter().all(|&x| x == 0x0101_0108));
        assert!(actual[n..].iter().all(|&x| x == 0x0101_0101));

        let other = Device::from_native(raw_device.clone())?;
        // SAFETY: the previous export completed all work; the old native view is no longer used.
        let mut adopted = unsafe { Buffer::<u32>::from_native(&other, raw, n + 32)? };
        assert!(matches!(
            // SAFETY: this context mismatch must be rejected before any encoding.
            unsafe {
                device.launch(
                    &pipeline,
                    LaunchConfig::<32>::for_elements(n as u32)?,
                    &[
                        Argument::read_write(&mut adopted),
                        Argument::value(n as u32)?,
                    ],
                )
            },
            Err(Error::DeviceMismatch)
        ));
        let other_module = Module::from_source(&other, SOURCE)?;
        let other_pipeline = Pipeline::new(&other, &other_module, "add")?;
        // SAFETY: the explicitly adopted context owns initialized, sufficiently sized storage.
        unsafe {
            other.launch(
                &other_pipeline,
                LaunchConfig::<32>::for_elements(n as u32)?,
                &[
                    Argument::read_write(&mut adopted),
                    Argument::value(n as u32)?,
                ],
            )?
        };
        assert!(adopted.as_slice()[..n].iter().all(|&x| x == 0x0101_010f));
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn native_import_checks_extent_and_storage_and_preserves_codecs() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let raw = device.buffer_zeroed::<u64>(2)?.into_native();
    // SAFETY: idle initialized storage; invalid extents are rejected before access.
    unsafe {
        assert!(matches!(
            Buffer::<u64>::from_native(&device, raw.clone(), 3),
            Err(Error::InvalidBuffer(_))
        ));
        assert!(matches!(
            Buffer::<u64>::from_native(&device, raw.clone(), usize::MAX),
            Err(Error::LengthOverflow)
        ));
        let empty = Buffer::<u64>::from_native(&device, raw, 0)?;
        assert!(empty.as_slice().is_empty());
    }
    let private = device
        .as_native()
        .newBufferWithLength_options(16, MTLResourceOptions::StorageModePrivate)
        .unwrap();
    assert!(matches!(
        // SAFETY: no work uses this allocation; a zero logical extent needs no initialized elements.
        unsafe { Buffer::<u32>::from_native(&device, private, 0) },
        Err(Error::InvalidBuffer(_))
    ));
    let untracked = device
        .as_native()
        .newBufferWithLength_options(
            16,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked,
        )
        .unwrap();
    assert!(matches!(
        // SAFETY: no work uses this allocation; validation rejects untracked native storage.
        unsafe { Buffer::<u32>::from_native(&device, untracked, 0) },
        Err(Error::InvalidBuffer(_))
    ));

    let initial = [(1_u8, 100_u32), (2, 200), (3, 300)];
    let raw = device.buffer_from_slice(&initial)?.into_native();
    let address = raw.contents().as_ptr();
    // SAFETY: export initialized the canonical tuple encoding and completed pending accesses.
    let mut values = unsafe { Buffer::<(u8, u32)>::from_native(&device, raw, initial.len())? };
    assert_ne!(
        values.as_slice().as_ptr().cast::<u8>(),
        address.cast::<u8>()
    );
    assert_eq!(values.as_slice(), initial);
    values.as_mut_slice()[1] = (4, 400);
    let raw = values.into_native();
    assert_eq!(raw.contents().as_ptr(), address);
    // SAFETY: the canonical encoding is initialized and idle after export.
    let encoded = unsafe { std::slice::from_raw_parts(raw.contents().cast::<u8>().as_ptr(), 24) };
    assert_eq!(encoded[8], 4);
    assert_eq!(&encoded[9..12], &[0; 3]);
    assert_eq!(&encoded[12..16], &400_u32.to_le_bytes());
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn native_storage_survives_discarded_batches_and_owned_resource_drop() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "add")?;
    let raw = device.buffer_from_slice(&[3_u32; 32])?.into_native();
    // SAFETY: initialized idle allocation; the observer alias remains idle until completion.
    let mut values = unsafe { Buffer::<u32>::from_native(&device, raw.clone(), 32)? };
    // SAFETY: the first dispatch is in bounds; returning an encoding error discards it.
    let error = unsafe {
        device.submit(|batch| {
            batch.launch(
                &pipeline,
                LaunchConfig::<32>::for_elements(32)?,
                &[Argument::read_write(&mut values), Argument::value(32_u32)?],
            )?;
            Err(Error::InvalidLaunch("discard batch"))
        })
    };
    assert!(matches!(error, Err(Error::InvalidLaunch("discard batch"))));
    assert_eq!(values.as_slice(), &[3; 32]);
    // SAFETY: the batch retains its owned imported buffer and pipeline after the closure drops them.
    let submission = unsafe {
        device.submit(move |batch| {
            batch.launch(
                &pipeline,
                LaunchConfig::<32>::for_elements(32)?,
                &[Argument::read_write(&mut values), Argument::value(32_u32)?],
            )
        })?
    };
    submission.wait()?;
    drop(module);
    drop(device);
    // SAFETY: all runtime accesses ended, and the observer still retains initialized native storage.
    let actual = unsafe { std::slice::from_raw_parts(raw.contents().cast::<u32>().as_ptr(), 32) };
    assert_eq!(actual, &[10; 32]);
    Ok(())
}
