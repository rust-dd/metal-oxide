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
    let mut first = new_batch(&device, &queue, &cache);
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
    let second = new_batch(&device, &queue, &cache);
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
    assert!(matches!(cache.poll_take(&mut context()), Poll::Ready(None)));
    completion.finish(Ok(None));
    let Poll::Ready(Some(reused)) = cache.poll_take(&mut context()) else {
        panic!("completed storage was not reusable");
    };
    assert!(std::ptr::eq(&*storage.allocator, &*reused.allocator));
    assert!(std::ptr::eq(&*storage.command, &*reused.command));
    assert!(std::ptr::eq(&*storage.residency, &*reused.residency));
    assert!(matches!(cache.poll_take(&mut context()), Poll::Ready(None)));
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
            let batch = new_batch(&device, &queue, &cache);
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
        let batch = new_batch(&device, &queue, &cache);
        assert!(
            commands
                .iter()
                .any(|command| std::ptr::eq(&**command, &*batch.command))
        );
    }
    assert!(matches!(cache.poll_take(&mut context()), Poll::Ready(None)));
}

#[test]
#[ignore = "requires a Metal 4 device on Apple Silicon"]
fn a_full_command_storage_pool_yields_and_wakes_on_completion() {
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
    let wake = Arc::new(WakeCount::default());
    let waker = std::task::Waker::from(Arc::clone(&wake));
    let mut cx = Context::from_waker(&waker);
    assert!(cache.poll_take(&mut cx).is_pending());
    assert_eq!(cache.entries().len(), MAX_BATCHES);
    assert_eq!(wake.0.load(std::sync::atomic::Ordering::Relaxed), 0);
    // A later batch can finish first on another queue.
    pending[1].1.finish(Ok(None));
    assert_eq!(wake.0.load(std::sync::atomic::Ordering::Relaxed), 1);
    let Poll::Ready(Some(reused)) = cache.poll_take(&mut cx) else {
        panic!("completed storage was not reusable");
    };
    assert!(std::ptr::eq(&*pending[1].0.allocator, &*reused.allocator));
    assert_eq!(cache.entries().len(), MAX_BATCHES - 1);
}

#[test]
#[ignore = "requires a Metal 4 device on Apple Silicon"]
fn cancelling_a_launch_waiting_for_storage_releases_its_buffer() -> Result<()> {
    use std::{
        future::{Future, IntoFuture},
        pin::Pin,
    };

    let device = crate::Device::system_default()?;
    let super::super::backend::Backend::Metal4 { cache, .. } = device.backend() else {
        return Ok(());
    };
    let module = crate::Module::from_source(
        &device,
        "#include <metal_stdlib>\nkernel void set(device uint *out [[buffer(0)]]) { out[0] = 42; }",
    )?;
    let pipeline = crate::Pipeline::new(&device, &module, "set")?;
    let pending = (0..MAX_BATCHES)
        .map(|_| {
            let completion = Arc::new(Completion::default());
            cache.put(
                CommandStorage::new(device.raw()).unwrap(),
                Arc::clone(&completion),
            );
            completion
        })
        .collect::<Vec<_>>();
    let mut output = device.buffer_zeroed::<u32>(1)?;
    let config = crate::LaunchConfig::<1>::for_elements(1)?;
    // SAFETY: one thread writes the single initialized output element.
    let operation = unsafe {
        crate::Launch::new(Ok(&pipeline), config, &mut output, |output| {
            Ok([Argument::write(output)])
        })
    };
    let mut future = operation.into_future();
    assert!(Pin::new(&mut future).poll(&mut context()).is_pending());
    drop(future);
    assert_eq!(output.read().wait(), [0]);
    pending[3].finish(Ok(None));
    // SAFETY: the same one-element dispatch now has available command storage.
    unsafe {
        crate::Launch::new(Ok(&pipeline), config, &mut output, |output| {
            Ok([Argument::write(output)])
        })
    }
    .wait()?;
    assert_eq!(output.read().wait(), [42]);
    Ok(())
}

#[derive(Default)]
struct WakeCount(std::sync::atomic::AtomicUsize);

impl std::task::Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

fn context() -> Context<'static> {
    Context::from_waker(std::task::Waker::noop())
}

fn new_batch(
    device: &crate::Device,
    queue: &Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    cache: &BatchCache,
) -> Metal4Batch {
    let Poll::Ready(batch) =
        Metal4Batch::poll_new(device.raw(), queue, cache, None, true, &mut context())
    else {
        panic!("test pool has capacity");
    };
    batch.unwrap()
}
