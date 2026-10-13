use std::ptr::NonNull;

use block2::RcBlock;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLDevice, MTLLogLevel, MTLLogState, MTLLogStateDescriptor};

use crate::{Error, Result};

const BUFFER_BYTES: isize = 1 << 20;

/// Prints the `log!` messages of a device's kernels to stderr; Metal delivers them
/// asynchronously around command completion.
pub(super) fn log_state(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLLogState>>> {
    let descriptor = MTLLogStateDescriptor::new();
    descriptor.setLevel(MTLLogLevel::Debug);
    // SAFETY: the size exceeds Metal's 1 KiB minimum.
    unsafe { descriptor.setBufferSize(BUFFER_BYTES) };
    // The Apple Paravirtual device of macOS VMs reports Mac2 but cannot create a log state.
    let state = device
        .newLogStateWithDescriptor_error(&descriptor)
        .map_err(|error| {
            Error::UnsupportedDevice(format!(
                "shader logging is unavailable: {}",
                error.localizedDescription()
            ))
        })?;
    let handler = RcBlock::new(
        |_: *mut NSString, _: *mut NSString, _: MTLLogLevel, message: NonNull<NSString>| {
            // SAFETY: Metal passes a live message string.
            eprintln!("{}", unsafe { message.as_ref() });
        },
    );
    // SAFETY: the handler captures nothing, so any thread may run it.
    unsafe { state.addLogHandler(&handler) };
    Ok(state)
}
