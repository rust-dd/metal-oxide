#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::{sync::mpsc, thread};

use metal_oxide::{Argument, Device, LaunchConfig, Module, Pipeline, Result};

struct Job {
    len: u32,
    initial: f32,
    reply: mpsc::Sender<Result<Vec<f32>>>,
}

fn worker(jobs: mpsc::Receiver<Job>) -> Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(
        &device,
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void affine(device float* data [[buffer(0)]], constant uint& n [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < n) data[i] = data[i] * 2.0f + 1.0f;
}
"#,
    )?;
    let pipeline = Pipeline::new(&device, &module, "affine")?;
    let mut values = device.buffer_zeroed::<f32>(257)?;
    for job in jobs {
        let result = (|| {
            if values.len() < job.len as usize {
                values = device.buffer_zeroed(job.len as usize)?;
            }
            values.as_mut_slice()[..job.len as usize].fill(job.initial);
            // SAFETY: the buffer covers the guarded 1D launch; both passes are ordered.
            unsafe {
                device.submit(|batch| {
                    for _ in 0..2 {
                        batch.launch(
                            &pipeline,
                            LaunchConfig::<32>::for_elements(job.len)?,
                            &[Argument::read_write(&mut values), Argument::value(job.len)?],
                        )?;
                    }
                    Ok(())
                })?
            }
            .wait()?;
            Ok(values.as_slice()[..job.len as usize].to_vec())
        })();
        // A cancelled requester does not stop the shared worker or discard its cache.
        let _ = job.reply.send(result);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn cached_worker_serves_multiple_threads_and_cancelled_requesters() {
    let (sender, receiver) = mpsc::channel();
    let worker = thread::spawn(move || worker(receiver));
    let clients = (0..4)
        .map(|client| {
            let sender = sender.clone();
            thread::spawn(move || {
                let (reply, cancelled) = mpsc::channel();
                drop(cancelled);
                sender
                    .send(Job {
                        len: 257,
                        initial: -10.0,
                        reply,
                    })
                    .unwrap();
                for len in [0, 1, 31, 32, 33, 257, 1025] {
                    let (reply, result) = mpsc::channel();
                    let initial = client as f32 + len as f32;
                    sender
                        .send(Job {
                            len,
                            initial,
                            reply,
                        })
                        .unwrap();
                    assert_eq!(
                        result.recv().unwrap().unwrap(),
                        vec![initial * 4.0 + 3.0; len as usize]
                    );
                }
            })
        })
        .collect::<Vec<_>>();
    drop(sender);
    for client in clients {
        client.join().unwrap();
    }
    worker.join().unwrap().unwrap();
}
