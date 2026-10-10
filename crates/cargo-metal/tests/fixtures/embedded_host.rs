#![deny(warnings)]

mod kernels {
    include!("../bindings.rs");
}

use kernels::load_bytes;

fn main() -> metal_oxide::Result<()> {
    use metal_oxide::{Argument, Device, Dim3, Error, LaunchConfig, Module, Pipeline};
    let device = Device::system_default()?;
    let manifest = include_str!("../manifest.json");
    let library = include_bytes!("../kernels.metallib");
    let start = std::time::Instant::now();
    let kernels = load_bytes(&device, manifest, library)?;
    println!("embedded load: {:?}", start.elapsed());
    let mut values = device.buffer_from_slice(&[0xffff_ffff_0000_0000_u64; 25])?;
    let config = LaunchConfig::<8>::for_elements(17)?;
    let start = std::time::Instant::now();
    // SAFETY: 17 distinct lanes update initialized elements; padding remains untouched.
    unsafe { kernels.in_place(config, &mut values, 7, 17)? };
    println!("first pipeline and dispatch: {:?}", start.elapsed());
    let start = std::time::Instant::now();
    for _ in 0..10 {
        // SAFETY: the same in-bounds launch reuses its initialized pipeline and buffer.
        unsafe { kernels.in_place(config, &mut values, 0, 17)? };
    }
    println!("ten cached dispatches: {:?}", start.elapsed());
    // SAFETY: ordered dispatches reuse the same exclusive buffer and stay within 17 elements.
    unsafe {
        device
            .submit(|batch| {
                kernels.enqueue_in_place(batch, config, &mut values, 9, 17)?;
                kernels.enqueue_in_place(batch, config, &mut values, 11, 17)
            })?
            .wait()?;
        kernels.k317(LaunchConfig::<1>::new(Dim3::x(1)))?;
    }
    assert_eq!(&values.as_slice()[..17], &[0xffff_ffff_0000_001b; 17]);
    assert_eq!(&values.as_slice()[17..], &[0xffff_ffff_0000_0000; 8]);

    // SAFETY: this entry has no arguments or memory accesses. Its missing native
    // function must fail on use, without preventing unrelated kernels from loading.
    let missing = unsafe { kernels.k482(LaunchConfig::<1>::new(Dim3::x(1))) };
    assert!(matches!(missing, Err(Error::KernelNotFound(name)) if name == "k482"));

    let module = Module::from_bytes(&device, manifest, library)?;
    assert!(module.verify_abi(&"0".repeat(64)).is_err());
    let input = device.buffer_from_slice(&[1.0_f32; 17])?;
    let mut reductions = device.buffer_zeroed::<f32>(3)?;
    // SAFETY: both scratch counts match the block width, all lanes initialize
    // their cells and participate, and three blocks own separate output elements.
    unsafe {
        kernels.dynamic_reduce(LaunchConfig::<8>::for_elements(17)?, &input, 8, 0.5, &mut reductions, 8, 17)?;
        device.submit(|batch| {
            kernels.enqueue_dynamic_reduce(batch, LaunchConfig::<8>::for_elements(17)?, &input, 8, 0.5, &mut reductions, 8, 17)
        })?.wait()?;
    }
    assert_eq!(reductions.as_slice(), [75.0, 203.0, 40.5]);
    let dynamic = Pipeline::new(&device, &module, "dynamic_reduce")?;
    // SAFETY: the empty grid accesses no memory; mismatched scratch types and
    // counts must still be rejected through the artifact contract.
    unsafe {
        for wrong in [Argument::threadgroup::<u32>(8)?, Argument::value(8_u32)?] {
            assert!(matches!(device.launch(&dynamic, LaunchConfig::<8>::new(Dim3::x(0)), &[
                Argument::read(&input), wrong, Argument::value(0.5_f32)?, Argument::write(&mut reductions),
                Argument::threadgroup::<(u8, u64)>(8)?, Argument::value(17_u32)?,
            ]), Err(Error::Artifact(_))));
        }
        assert!(matches!(device.launch(&dynamic, LaunchConfig::<8>::new(Dim3::x(0)), &[]), Err(Error::Artifact(_))));
    }
    let pipeline = Pipeline::new(&device, &module, "in_place")?;
    // SAFETY: empty grids cannot access memory; invalid access, scalar layout and
    // specialization must still be rejected before the empty-launch fast path.
    unsafe {
        for access in [false, true] {
            let argument = if access {
                Argument::write(&mut values)
            } else {
                Argument::read(&values)
            };
            assert!(matches!(
                device.launch(
                    &pipeline,
                    LaunchConfig::<8>::new(Dim3::x(0)),
                    &[argument, Argument::value(1_u64)?, Argument::value(17_u32)?,]
                ),
                Err(Error::Artifact(_))
            ));
        }
        assert!(matches!(
            device.launch(
                &pipeline,
                LaunchConfig::<8>::new(Dim3::x(0)),
                &[
                    Argument::read_write(&mut values),
                    Argument::value(1_u32)?,
                    Argument::value(17_u32)?,
                ]
            ),
            Err(Error::Artifact(_))
        ));
        assert!(matches!(
            device.launch(
                &pipeline,
                LaunchConfig::<32>::new(Dim3::x(0)),
                &[
                    Argument::read_write(&mut values),
                    Argument::value(1_u64)?,
                    Argument::value(17_u32)?,
                ]
            ),
            Err(Error::InvalidLaunch(_))
        ));
    }
    println!("embedded consumer passed");
    Ok(())
}
