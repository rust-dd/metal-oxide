use super::*;

#[test]
#[ignore = "requires a Metal 4 device on Apple Silicon"]
fn reused_tables_follow_changing_kernel_bindings() -> Result<()> {
    let device = crate::Device::system_default()?;
    let super::super::backend::Backend::Metal4 { .. } = device.backend() else {
        return Ok(());
    };
    let module = crate::Module::from_source(
        &device,
        r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void short_args(device uint *out [[buffer(0)]],
            constant uint &value [[buffer(1)]]) { out[0] = value; }
        kernel void long_args(device uint *out [[buffer(0)]],
            constant uint &a [[buffer(1)]], constant uint &b [[buffer(2)]],
            constant uint &c [[buffer(3)]]) { out[0] = a + b + c; }
        "#,
    )?;
    let short = crate::Pipeline::new(&device, &module, "short_args")?;
    let long = crate::Pipeline::new(&device, &module, "long_args")?;
    let mut output = device.buffer_zeroed::<u32>(1)?;
    for (pipeline, values) in [
        (&short, &[11_u32][..]),
        (&long, &[2, 3, 4][..]),
        (&short, &[17][..]),
        (&long, &[5, 6, 7][..]),
    ] {
        let mut arguments = vec![Argument::write(&mut output)];
        for &value in values {
            arguments.push(Argument::value(value)?);
        }
        // SAFETY: each kernel's initialized constants match its signature; one thread
        // writes the single output. Waiting ends all accesses before the next dispatch.
        unsafe {
            device.launch(
                pipeline,
                crate::LaunchConfig::<1>::for_elements(1)?,
                &arguments,
            )?;
        }
        drop(arguments);
        assert_eq!(output.as_slice(), [values.iter().sum::<u32>()]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal 4 device on Apple Silicon"]
fn value_cache_bounds_retention_across_large_and_small_dispatches() -> Result<()> {
    let device = crate::Device::system_default()?;
    let super::super::backend::Backend::Metal4 { cache, .. } = device.backend() else {
        return Ok(());
    };
    let module = crate::Module::from_source(
        &device,
        r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void values(device ulong *out [[buffer(0)]],
            constant uint *values [[buffer(1)]], constant uint &count [[buffer(2)]],
            constant ulong &tag [[buffer(3)]]) {
            out[0] = ulong(values[0]) + values[count - 1] + tag;
        }
        "#,
    )?;
    let pipeline = crate::Pipeline::new(&device, &module, "values")?;
    let mut output = device.buffer_zeroed::<u64>(1)?;
    for (values, count, retained) in [
        (Argument::value([7_u32; 1022])?, 1022_u32, 0),
        (Argument::value([7_u32; 1020])?, 1020_u32, 1),
        (Argument::value(7_u32)?, 1_u32, 1),
    ] {
        // SAFETY: one thread writes one output; the initialized array covers count.
        unsafe {
            device.submit(|batch| {
                batch.launch(
                    &pipeline,
                    crate::LaunchConfig::<1>::for_elements(1)?,
                    &[
                        Argument::write(&mut output),
                        values,
                        Argument::value(count)?,
                        Argument::value(1_u64 << 40)?,
                    ],
                )
            })?
        }
        .wait()?;
        assert_eq!(output.as_slice(), [14 + (1_u64 << 40)]);
        let entry = cache.entry();
        let (storage, _) = entry.as_ref().unwrap();
        assert_eq!(storage.constants.len(), retained);
        assert!(
            storage
                .constants
                .iter()
                .all(|buffer| buffer.length() <= 4096)
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal 4 device on Apple Silicon"]
fn completed_batches_reuse_command_storage_without_stale_residency() {
    let device = crate::Device::system_default().unwrap();
    if !super::super::backend::Backend::supports_metal4(device.raw()) {
        return;
    }
    let queue = device.raw().newMTL4CommandQueue().unwrap();
    let cache = BatchCache::default();
    let mut first = Metal4Batch::new(device.raw(), &queue, &cache).unwrap();
    let command = first.command.clone();
    let residency = first.residency.clone();
    let buffer = super::super::constants::allocate(device.raw(), 16).unwrap();
    residency.addAllocation(ProtocolObject::from_ref(&*buffer));
    first.bindings.push(buffer);
    let completion = Arc::new(Completion::default());
    first.commit(Arc::clone(&completion), Vec::new());
    super::super::Submission {
        completion,
        marker: std::marker::PhantomData,
    }
    .wait()
    .unwrap();
    let second = Metal4Batch::new(device.raw(), &queue, &cache).unwrap();
    assert!(std::ptr::eq(&*command, &*second.command));
    assert!(std::ptr::eq(&*residency, &*second.residency));
    assert_eq!(second.residency.allocationCount(), 0);
}

#[test]
#[ignore = "requires a Metal 4 device on Apple Silicon"]
fn command_storage_reuse_requires_completed_gpu_work() {
    let device = crate::Device::system_default().unwrap();
    if !super::super::backend::Backend::supports_metal4(device.raw()) {
        return;
    }
    let storage = CommandStorage::new(device.raw()).unwrap();
    let completion = Arc::new(Completion::default());
    let cache = BatchCache::default();
    *cache.entry() = Some((storage.clone(), Arc::clone(&completion)));
    assert!(cache.take().is_none());
    completion.finish(Ok(SubmissionReport::default()));
    *cache.entry() = Some((storage.clone(), completion));
    let reused = cache.take().unwrap();
    assert!(std::ptr::eq(&*storage.allocator, &*reused.allocator));
    assert!(std::ptr::eq(&*storage.command, &*reused.command));
    assert!(std::ptr::eq(&*storage.residency, &*reused.residency));
    assert!(cache.take().is_none());
}
