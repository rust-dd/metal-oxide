use super::*;
use crate::{Argument, Operation, metal::queue::Finish};

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
    let config = crate::LaunchConfig::<1>::for_elements(1)?;
    for (pipeline, values) in [
        (&short, &[11_u32][..]),
        (&long, &[2, 3, 4][..]),
        (&short, &[17][..]),
        (&long, &[5, 6, 7][..]),
    ] {
        // SAFETY: each kernel's initialized constants match its signature; one thread
        // writes the single output. Waiting ends all accesses before the next dispatch.
        unsafe {
            match *values {
                [value] => crate::Launch::new(
                    Ok(pipeline),
                    config,
                    (&mut output, value),
                    |(output, value)| Ok([Argument::write(output), Argument::value(*value)?]),
                )
                .wait()
                .map(drop),
                [a, b, c] => crate::Launch::new(
                    Ok(pipeline),
                    config,
                    (&mut output, [a, b, c]),
                    |(output, [a, b, c])| {
                        Ok([
                            Argument::write(output),
                            Argument::value(*a)?,
                            Argument::value(*b)?,
                            Argument::value(*c)?,
                        ])
                    },
                )
                .wait()
                .map(drop),
                _ => unreachable!(),
            }?;
        }
        assert_eq!(output.read().wait(), [values.iter().sum::<u32>()]);
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
    /// One thread writes one output; the initialized array covers count.
    fn run<V: crate::GpuValue>(
        pipeline: &crate::Pipeline,
        output: &mut crate::Buffer<u64>,
        values: V,
        count: u32,
    ) -> Result<()> {
        let config = crate::LaunchConfig::<1>::for_elements(1)?;
        // SAFETY: see above.
        unsafe {
            crate::Launch::new(
                Ok(pipeline),
                config,
                (output, values, count),
                |(output, values, count)| {
                    Ok([
                        Argument::write(output),
                        Argument::value(*values)?,
                        Argument::value(*count)?,
                        Argument::value(1_u64 << 40)?,
                    ])
                },
            )
            .wait()
            .map(drop)
        }
    }
    for (count, retained) in [(1022_u32, 0), (1020, 1), (1, 1)] {
        match count {
            1022 => run(&pipeline, &mut output, [7_u32; 1022], count)?,
            1020 => run(&pipeline, &mut output, [7_u32; 1020], count)?,
            _ => run(&pipeline, &mut output, 7_u32, count)?,
        }
        assert_eq!(output.read().wait(), [14 + (1_u64 << 40)]);
        let entries = cache.entries();
        assert_eq!(entries.len(), 1);
        let (storage, _) = &entries[0];
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
    let mut first = Metal4Batch::new(device.raw(), &queue, &cache, None, true).unwrap();
    let command = first.command.clone();
    let residency = first.residency.clone();
    let buffer = super::super::constants::allocate(device.raw(), 16).unwrap();
    residency.addAllocation(ProtocolObject::from_ref(&*buffer));
    first.bindings.push(buffer);
    let completion = Arc::new(Completion::default());
    first.commit(
        &cache,
        Arc::clone(&completion),
        Vec::new(),
        Arc::new(Failures::new(Arc::clone(device.failure_pool()))),
        &[],
        Finish::new(Arc::new(std::sync::atomic::AtomicUsize::new(1)), None),
    );
    completion.wait().unwrap();
    let second = Metal4Batch::new(device.raw(), &queue, &cache, None, true).unwrap();
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
    cache.put(storage.clone(), Arc::clone(&completion));
    assert!(cache.take().is_none());
    completion.finish(Ok(None));
    let reused = cache.take().unwrap();
    assert!(std::ptr::eq(&*storage.allocator, &*reused.allocator));
    assert!(std::ptr::eq(&*storage.command, &*reused.command));
    assert!(std::ptr::eq(&*storage.residency, &*reused.residency));
    assert!(cache.take().is_none());
}

#[test]
#[ignore = "requires a Metal 4 device on Apple Silicon"]
fn pipelined_batches_reuse_pooled_command_storage() {
    let device = crate::Device::system_default().unwrap();
    if !super::super::backend::Backend::supports_metal4(device.raw()) {
        return;
    }
    let queue = device.raw().newMTL4CommandQueue().unwrap();
    let cache = BatchCache::default();
    let mut commands = Vec::new();
    let completions = (0..3)
        .map(|_| {
            let batch = Metal4Batch::new(device.raw(), &queue, &cache, None, true).unwrap();
            commands.push(batch.command.clone());
            let completion = Arc::new(Completion::default());
            batch.commit(
                &cache,
                Arc::clone(&completion),
                Vec::new(),
                Arc::new(Failures::new(Arc::clone(device.failure_pool()))),
                &[],
                Finish::new(Arc::new(std::sync::atomic::AtomicUsize::new(1)), None),
            );
            completion
        })
        .collect::<Vec<_>>();
    assert!(!std::ptr::eq(&*commands[0], &*commands[1]));
    completions.iter().for_each(|completion| completion.block());
    for _ in 0..3 {
        let batch = Metal4Batch::new(device.raw(), &queue, &cache, None, true).unwrap();
        assert!(
            commands
                .iter()
                .any(|command| std::ptr::eq(&**command, &*batch.command))
        );
    }
    assert!(cache.take().is_none());
}

#[test]
#[ignore = "requires a Metal 4 device on Apple Silicon"]
fn a_full_command_storage_pool_waits_for_its_oldest_batch() {
    let device = crate::Device::system_default().unwrap();
    if !super::super::backend::Backend::supports_metal4(device.raw()) {
        return;
    }
    let cache = BatchCache::default();
    let pending = (0..MAX_BATCHES)
        .map(|_| {
            let storage = CommandStorage::new(device.raw()).unwrap();
            let completion = Arc::new(Completion::default());
            cache.put(storage.clone(), Arc::clone(&completion));
            (storage, completion)
        })
        .collect::<Vec<_>>();
    let oldest = Arc::clone(&pending[0].1);
    let finisher = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(20));
        oldest.finish(Ok(None));
    });
    let reused = cache.take().unwrap();
    finisher.join().unwrap();
    assert!(std::ptr::eq(&*pending[0].0.allocator, &*reused.allocator));
    assert_eq!(cache.entries().len(), MAX_BATCHES - 1);
}
