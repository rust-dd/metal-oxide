#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::sync::{Arc, Barrier};

use metal_oxide::{Argument, Device, LaunchConfig, Module, Pipeline};

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn independent_contexts_compile_and_run_the_same_library_concurrently() {
    let source = (0..4).map(|index| format!(
        "kernel void fill_{index}(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{ out[i] = i + {index}; }}\n"
    )).collect::<String>();
    let source = format!("#include <metal_stdlib>\nusing namespace metal;\n{source}");
    let barrier = Arc::new(Barrier::new(4));
    let workers = (0..4)
        .map(|_| {
            let barrier = barrier.clone();
            let source = source.clone();
            std::thread::spawn(move || -> metal_oxide::Result<()> {
                let prepared = Device::system_default().and_then(|device| {
                    let module = Module::from_source(&device, &source)?;
                    Ok((device, module))
                });
                barrier.wait();
                let (device, module) = prepared?;
                for index in 0..4 {
                    let pipeline = Pipeline::new(&device, &module, &format!("fill_{index}"))?;
                    let mut output = device.buffer_zeroed::<u32>(32)?;
                    // SAFETY: each lane writes its own initialized output element.
                    unsafe {
                        device.launch(
                            &pipeline,
                            LaunchConfig::<32>::for_elements(32)?,
                            &[Argument::write(&mut output)],
                        )?;
                    }
                    assert_eq!(
                        output.as_slice(),
                        (0..32).map(|i| i + index).collect::<Vec<_>>()
                    );
                }
                Ok(())
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap().unwrap();
    }
}
