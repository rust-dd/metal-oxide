#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use metal_oxide::{
    Argument, Batch, Buffer, Device, Dim3, Error, GpuValue, Launch, LaunchConfig, Layout, Library,
    Module, Operation, Pipeline, Result, sequence,
};

#[derive(Clone, Copy)]
struct Value(u32);

impl GpuValue for Value {
    const SIZE: usize = 4;
    const ALIGNMENT: usize = 4;

    fn layout() -> Result<Layout> {
        u32::layout()
    }
    fn zeroed() -> Self {
        Self(0)
    }
    fn encode(self, bytes: &mut [u8]) {
        self.0.encode(bytes);
    }
    fn decode(bytes: &[u8]) -> Self {
        Self(u32::decode(bytes))
    }
}

#[test]
fn debug_does_not_require_argument_or_element_debug() {
    fn debug<T: std::fmt::Debug>() {}
    debug::<Device>();
    debug::<Buffer<Value>>();
    debug::<Module>();
    debug::<Pipeline>();
    debug::<Launch<Value, 0>>();
    debug::<Batch>();
}

#[test]
fn formatting_an_unstarted_launch_does_not_bind_its_arguments() {
    // SAFETY: the deferred error prevents the launch from ever binding or dispatching.
    let launch = unsafe {
        Launch::<_, 0>::new(
            Err(Error::DeviceUnavailable),
            LaunchConfig::<32>::new(Dim3::x(7)),
            Value(1),
            |_| panic!("formatting must not bind arguments"),
        )
    };
    let text = format!("{launch:?}");
    assert!(text.starts_with("Launch {"), "{text}");
    assert!(text.contains("DeviceUnavailable"), "{text}");
    assert!(text.contains("grid: Dim3 { x: 7"), "{text}");
    assert!(text.contains("arguments: 0"), "{text}");
}

#[test]
fn an_empty_batch_reports_its_recording_counts() -> Result<()> {
    sequence(|batch| {
        let text = format!("{batch:?}");
        assert!(text.contains("device_id: None"), "{text}");
        assert!(text.contains("dispatches: 0"), "{text}");
        assert!(text.contains("bindings: 0"), "{text}");
        Ok(())
    })
    .wait()
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn formatting_handles_keeps_buffer_contents_and_recordings_opaque() -> Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(
        &device,
        "#include <metal_stdlib>\nkernel void set(device uint* out [[buffer(0)]]) { out[0] = 42; }",
    )?;
    let pipeline = Pipeline::new(&device, &module, "set")?;
    let mut buffer = device.buffer_zeroed::<Value>(65_537)?;
    for (name, text) in [
        ("Device", format!("{device:?}")),
        ("Module", format!("{module:?}")),
        ("Pipeline", format!("{pipeline:?}")),
        ("Buffer", format!("{buffer:?}")),
    ] {
        assert!(text.starts_with(name), "{text}");
        assert!(text.len() < 512, "{text}");
    }
    assert!(format!("{pipeline:?}").contains("entrypoint: \"set\""));
    let before = format!("{buffer:?}");
    assert!(before.contains("len: 65537"), "{before}");
    sequence(|batch| {
        // SAFETY: one thread writes the first of the buffer's initialized u32 values.
        let launch = unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<1>::new(Dim3::x(1)),
                &mut buffer,
                |buffer| Ok([Argument::write(buffer)]),
            )
        };
        assert!(format!("{launch:?}").contains("entrypoint: \"set\""));
        let buffer = batch.push(launch)?;
        assert_eq!(format!("{buffer:?}"), before);
        let text = format!("{batch:?}");
        assert!(text.contains("dispatches: 1"), "{text}");
        assert!(text.contains("bindings: 1"), "{text}");
        Ok(())
    })
    .wait()?;
    assert_eq!(format!("{buffer:?}"), before);
    assert_eq!(buffer.read().wait()[0].0, 42);

    static LIBRARY: Library = Library::new("not loaded while formatting", &[], "");
    // SAFETY: the empty launch is never submitted, and formatting must leave it unprepared.
    let launch = unsafe {
        Launch::kernel(
            &device,
            &LIBRARY,
            "unprepared",
            LaunchConfig::<1>::new(Dim3::x(0)),
            (),
            |_| Ok([]),
        )
    };
    let text = format!("{launch:?}");
    assert!(text.contains("entrypoint: \"unprepared\""), "{text}");
    assert!(text.contains("preparing: false"), "{text}");
    Ok(())
}
