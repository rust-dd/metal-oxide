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
    let state = device
        .newLogStateWithDescriptor_error(&descriptor)
        .map_err(|error| Error::Command(error.localizedDescription().to_string()))?;
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
