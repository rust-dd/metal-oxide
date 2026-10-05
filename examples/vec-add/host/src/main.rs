#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use metal_oxide::{Argument, Device, Dispatch1d, Module, Pipeline};

    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let device = Device::system_default()?;
    let module = match arguments.as_slice() {
        [] => Module::from_source(&device, include_str!("../../kernels/vec_add.metal"))?,
        [flag, path] if flag == "--metallib" => Module::from_metallib(&device, path)?,
        _ => return Err("usage: vec-add [--metallib PATH]".into()),
    };
    let pipeline = Pipeline::new(&device, &module, "vec_add")?;
    println!("Device: {}", device.name());
    println!(
        "Pipeline: execution width {}, max group size {}",
        pipeline.thread_execution_width(),
        pipeline.max_threads_per_threadgroup()
    );

    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        let a = (0..n).map(|i| (i % 1024) as f32 * 0.25).collect::<Vec<_>>();
        let b = (0..n).map(|i| (i % 511) as f32 * -0.5).collect::<Vec<_>>();
        let input_a = device.buffer_from_slice(&a)?;
        let input_b = device.buffer_from_slice(&b)?;
        let mut output = device.buffer_zeroed::<f32>(n as usize)?;

        // SAFETY: the reference vec_add ABI uses distinct n-element buffers and one writer per index.
        unsafe {
            device.dispatch(
                &pipeline,
                Dispatch1d::new(n),
                &[
                    Argument::read(&input_a),
                    Argument::read(&input_b),
                    Argument::write(&mut output),
                    Argument::u32(n),
                ],
            )?;
        }

        for (i, actual) in output.as_slice().iter().enumerate() {
            if *actual != a[i] + b[i] {
                return Err(format!("vec_add mismatch at n={n}, index={i}").into());
            }
        }
        println!("vec_add n={n}: verified");
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() -> std::process::ExitCode {
    eprintln!("vec-add requires macOS on Apple Silicon");
    std::process::ExitCode::FAILURE
}
