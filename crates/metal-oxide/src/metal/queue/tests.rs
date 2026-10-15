use super::*;

fn conflict(queue: usize, write: bool) -> Conflict {
    Conflict {
        queue,
        write,
        signal: None,
        failures: None,
    }
}

#[test]
fn dependent_batches_join_the_queue_of_their_conflicts() {
    let placement = Placement::new(&[3, 0, 0, 0], &[conflict(2, true), conflict(2, false)]);
    assert_eq!((placement.queue, placement.local), (2, true));
    assert!(placement.remote.is_empty());
    let placement = Placement::new(
        &[0, 0, 0, 0],
        &[conflict(1, true), conflict(3, false), conflict(3, true)],
    );
    assert_eq!((placement.queue, placement.local), (3, true));
    assert_eq!(
        placement
            .remote
            .iter()
            .map(|access| access.queue)
            .collect::<Vec<_>>(),
        [1]
    );
}

#[test]
fn independent_batches_go_to_the_least_busy_queue() {
    let placement = Placement::new(&[2, 1, 0, 1], &[]);
    assert_eq!((placement.queue, placement.local), (2, false));
    assert!(placement.remote.is_empty());
    assert_eq!(Placement::new(&[0, 0, 0, 0], &[]).queue, 0);
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn independent_operations_run_on_separate_queues() -> crate::Result<()> {
    use crate::{Argument, Launch, LaunchConfig, Module, Pipeline};
    use std::{future::IntoFuture, pin::Pin, task::Context};

    let device = crate::Device::system_default()?;
    let module = Module::from_source(
        &device,
        "#include <metal_stdlib>
        using namespace metal;
        kernel void spin(device float* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
            float x = data[i];
            for (uint k = 0; k < 2000000; ++k) { x = fma(x, 0.999f, 0.001f); }
            data[i] = x;
        }",
    )?;
    let pipeline = Pipeline::new(&device, &module, "spin")?;
    let mut buffers = [
        device.buffer_zeroed::<f32>(256)?,
        device.buffer_zeroed(256)?,
    ];
    let config = LaunchConfig::<256>::for_elements(256)?;
    let mut running = Vec::new();
    for buffer in &mut buffers {
        // SAFETY: each thread updates its own element of a 256-element buffer.
        let mut future = unsafe {
            Launch::new(Ok(&pipeline), config, (buffer,), |(buffer,)| {
                Ok([Argument::read_write(buffer)])
            })
        }
        .into_future();
        let _ = Pin::new(&mut future).poll(&mut Context::from_waker(std::task::Waker::noop()));
        running.push(device.backend().running());
        drop(future);
    }
    assert_eq!(running[0].iter().sum::<usize>(), 1);
    assert_eq!(
        running[1].iter().filter(|count| **count == 1).count(),
        2,
        "{running:?}"
    );
    Ok(())
}
