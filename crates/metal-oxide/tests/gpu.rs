#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use metal_oxide::{Argument, Device, Dim3, DynamicLaunchConfig, LaunchConfig, Module, Pipeline};

const SOURCE: &str = include_str!("../../../examples/vec-add/kernels/vec_add.metal");
const GEOMETRY_SOURCE: &str = include_str!("kernels/launch_geometry.metal");

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn vec_add_matches_cpu_and_preserves_padding() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "vec_add")?;

    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        verify_vec_add(&device, &pipeline, n, LaunchConfig::<32>::for_elements(n)?)?;
        verify_vec_add(&device, &pipeline, n, LaunchConfig::<256>::for_elements(n)?)?;
        verify_vec_add(
            &device,
            &pipeline,
            n,
            DynamicLaunchConfig::for_elements(n, pipeline.thread_execution_width() as u32)?,
        )?;
    }
    Ok(())
}

fn verify_vec_add(
    device: &Device,
    pipeline: &Pipeline,
    n: u32,
    config: impl Into<DynamicLaunchConfig> + Copy,
) -> metal_oxide::Result<()> {
    let block_size = config.into().block.x;
    let storage_len = n as usize + block_size as usize;
    let a = (0..storage_len)
        .map(|i| (i % 1024) as f32 * 0.25)
        .collect::<Vec<_>>();
    let b = (0..storage_len)
        .map(|i| (i % 511) as f32 * -0.5)
        .collect::<Vec<_>>();
    let input_a = device.buffer_from_slice(&a)?;
    let input_b = device.buffer_from_slice(&b)?;
    let sentinel = -8192.0_f32;
    let mut output = device.buffer_from_slice(&vec![sentinel; storage_len])?;

    // SAFETY: distinct allocations cover all launched threads; vec_add guards writes at n.
    unsafe {
        device.launch(
            pipeline,
            config,
            &[
                Argument::read(&input_a),
                Argument::read(&input_b),
                Argument::write(&mut output),
                Argument::value::<u32>(n)?,
            ],
        )?;
    }

    for (i, actual) in output.as_slice()[..n as usize].iter().enumerate() {
        assert_eq!(*actual, a[i] + b[i], "n={n}, block={block_size}, index={i}");
    }
    assert!(
        output.as_slice()[n as usize..]
            .iter()
            .all(|value| *value == sentinel),
        "padding was written at n={n}, block={block_size}"
    );
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn launch_geometry_matches_cuda_indexing_in_all_dimensions() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, GEOMETRY_SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "launch_geometry")?;

    verify_geometry(&device, &pipeline, LaunchConfig::<5>::new(Dim3::x(3)))?;
    verify_geometry(
        &device,
        &pipeline,
        LaunchConfig::<3, 2>::new(Dim3::xy(2, 3)),
    )?;
    verify_geometry(
        &device,
        &pipeline,
        LaunchConfig::<2, 3, 2>::new(Dim3::new(2, 2, 3)),
    )?;
    verify_geometry(
        &device,
        &pipeline,
        DynamicLaunchConfig::new(
            Dim3::xy(2, 3),
            Dim3::x(pipeline.thread_execution_width() as u32),
        ),
    )?;
    Ok(())
}

fn verify_geometry(
    device: &Device,
    pipeline: &Pipeline,
    config: impl Into<DynamicLaunchConfig> + Copy,
) -> metal_oxide::Result<()> {
    let geometry = config.into();
    let mut output = device.buffer_zeroed::<u32>(geometry.total_threads()? as usize * 15)?;

    // SAFETY: each global index owns one 15-word record in the fully sized output allocation.
    unsafe { device.launch(pipeline, config, &[Argument::write(&mut output)])? };

    let width = geometry.grid.x * geometry.block.x;
    let height = geometry.grid.y * geometry.block.y;
    for (index, record) in output.as_slice().as_chunks::<15>().0.iter().enumerate() {
        let x = index as u32 % width;
        let y = index as u32 / width % height;
        let z = index as u32 / (width * height);
        let expected = [
            x,
            y,
            z,
            x % geometry.block.x,
            y % geometry.block.y,
            z % geometry.block.z,
            x / geometry.block.x,
            y / geometry.block.y,
            z / geometry.block.z,
            geometry.block.x,
            geometry.block.y,
            geometry.block.z,
            geometry.grid.x,
            geometry.grid.y,
            geometry.grid.z,
        ];
        assert_eq!(*record, expected, "config={geometry:?}, index={index}");
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn empty_grid_leaves_output_untouched_on_every_axis() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, GEOMETRY_SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "launch_geometry")?;
    let sentinel = [0xdead_beef_u32; 15];

    for grid in [Dim3::new(0, 2, 2), Dim3::new(2, 0, 2), Dim3::new(2, 2, 0)] {
        let mut output = device.buffer_from_slice(&sentinel)?;
        let config = LaunchConfig::<2, 2, 2>::new(grid);
        // SAFETY: an empty grid executes no shader invocation.
        unsafe { device.launch(&pipeline, config, &[Argument::write(&mut output)])? };
        assert_eq!(output.as_slice(), sentinel);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn typed_buffers_are_initialized_and_mutable_on_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let mut buffer = device.buffer_zeroed::<u32>(3)?;
    assert_eq!(buffer.as_slice(), [0, 0, 0]);
    buffer.as_mut_slice().copy_from_slice(&[1, 2, 3]);
    assert_eq!(buffer.as_slice(), [1, 2, 3]);
    assert!(device.buffer_zeroed::<i32>(0)?.is_empty());
    assert!(device.buffer_zeroed::<u32>(usize::MAX).is_err());
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn invalid_shader_and_missing_entrypoint_return_errors() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    assert!(Module::from_source(&device, "this is not Metal source").is_err());
    let module = Module::from_source(&device, SOURCE)?;
    assert!(Pipeline::new(&device, &module, "missing").is_err());
    Ok(())
}
