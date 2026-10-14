use std::{
    ffi::OsStr,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

use objc2::{rc::autoreleasepool, runtime::ProtocolObject};
use objc2_foundation::NSURL;
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager, MTLDevice};

use crate::{Error, Result};

/// Set once a device of this process starts the capture; later devices do not record.
static STARTED: AtomicBool = AtomicBool::new(false);

/// Starts a GPU trace of `device` into the file `METAL_OXIDE_CAPTURE` names, unless it is unset
/// or another device started one; returns whether the trace started.
pub(super) fn start(device: &ProtocolObject<dyn MTLDevice>) -> Result<bool> {
    let Some(path) = std::env::var_os("METAL_OXIDE_CAPTURE") else {
        return Ok(false);
    };
    if STARTED.swap(true, Ordering::AcqRel) {
        return Ok(false);
    }
    let started = record(device, Path::new(&path));
    if started.is_err() {
        STARTED.store(false, Ordering::Release);
    }
    started.map(|()| true)
}

fn record(device: &ProtocolObject<dyn MTLDevice>, path: &Path) -> Result<()> {
    let failed = |reason: String| Error::Capture(format!("{}: {reason}", path.display()));
    if path.extension() != Some(OsStr::new("gputrace")) {
        return Err(failed("the file name must end in .gputrace".into()));
    }
    // SAFETY: the process's capture manager takes no arguments; Metal synchronizes it.
    let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
    if !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
        return Err(failed(
            "Metal writes GPU trace files only with METAL_CAPTURE_ENABLED=1 set at process start"
                .into(),
        ));
    }
    let absolute = std::path::absolute(path).map_err(|error| failed(error.to_string()))?;
    let url = NSURL::from_file_path(&absolute).ok_or_else(|| failed("invalid file path".into()))?;
    let descriptor = MTLCaptureDescriptor::new();
    // SAFETY: a device is a capture object; capturing it records all of its command queues.
    unsafe { descriptor.setCaptureObject(Some(device.as_ref())) };
    descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
    descriptor.setOutputURL(Some(&url));
    manager
        .startCaptureWithDescriptor_error(&descriptor)
        .map_err(|error| failed(error.localizedDescription().to_string()))
}

/// Ends the capture; Metal writes the command buffers committed before this call.
pub(super) fn stop() {
    // SAFETY: as in `record`.
    autoreleasepool(|_| unsafe { MTLCaptureManager::sharedCaptureManager() }.stopCapture());
}
