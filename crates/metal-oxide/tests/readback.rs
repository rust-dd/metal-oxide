#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::{
    cell::{Cell, RefCell},
    panic::{AssertUnwindSafe, catch_unwind},
    rc::Rc,
};

use metal_oxide::{Argument, Buffer, Device, GpuValue, LaunchConfig, Layout, Module, Pipeline};

thread_local! {
    static REENTER: RefCell<Option<Rc<Buffer<Value>>>> = const { RefCell::new(None) };
    static PANIC_AFTER: Cell<Option<usize>> = const { Cell::new(None) };
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Value(u32);

impl GpuValue for Value {
    const SIZE: usize = 4;
    const ALIGNMENT: usize = 4;
    fn layout() -> metal_oxide::Result<Layout> {
        u32::layout()
    }
    fn zeroed() -> Self {
        Self(0)
    }
    fn encode(self, bytes: &mut [u8]) {
        self.0.encode(bytes);
    }
    fn decode(bytes: &[u8]) -> Self {
        let nested = REENTER.with(|state| state.borrow_mut().take());
        if let Some(buffer) = nested {
            let _ = buffer.as_slice();
        }
        PANIC_AFTER.with(|state| match state.get() {
            Some(0) => {
                state.set(None);
                panic!("decoder failure");
            }
            Some(remaining) => state.set(Some(remaining - 1)),
            None => {}
        });
        Self(u32::decode(bytes))
    }
}

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void write_values(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = 10 + i;
}
"#;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn decoder_cannot_reenter_the_same_buffer() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "write_values")?;
    let mut out = device.buffer_zeroed::<Value>(3)?;
    // SAFETY: three threads write distinct initialized u32 values in the three-element buffer.
    unsafe {
        device.launch(
            &pipeline,
            LaunchConfig::<1>::for_elements(3)?,
            &[Argument::write(&mut out)],
        )?;
    }
    let out = Rc::new(out);
    REENTER.with(|state| *state.borrow_mut() = Some(Rc::clone(&out)));
    let result = catch_unwind(AssertUnwindSafe(|| out.as_slice()));
    assert!(result.is_err(), "recursive readback must be rejected");
    assert_eq!(out.as_slice(), [Value(10), Value(11), Value(12)]);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn decoder_panic_retries_readback_after_a_committed_write() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "write_values")?;
    let mut out = device.buffer_zeroed::<Value>(3)?;
    // SAFETY: CPU inspection precedes commit; three threads write distinct initialized elements.
    unsafe {
        device.submit(|batch| {
            batch.launch(
                &pipeline,
                LaunchConfig::<1>::for_elements(3)?,
                &[Argument::write(&mut out)],
            )?;
            assert_eq!(out.as_slice(), [Value(0); 3]);
            Ok(())
        })?
    }
    .wait()?;
    PANIC_AFTER.with(|state| state.set(Some(1)));
    assert!(catch_unwind(AssertUnwindSafe(|| out.as_slice())).is_err());
    assert_eq!(out.as_slice(), [Value(10), Value(11), Value(12)]);
    Ok(())
}
