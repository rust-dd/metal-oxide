use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicU64, Ordering},
};

use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{
    MTLComputePipelineState, MTLCopyAllDevices, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLogState,
};

use crate::{Dim3, Error, GpuValue, Result};

use super::{
    Buffer, Library, Module, Pipeline,
    backend::Backend,
    failures::{FailurePool, SNAPSHOT_KERNEL, SNAPSHOT_SOURCE},
    library::Libraries,
};

static NEXT_DEVICE_ID: AtomicU64 = AtomicU64::new(1);

/// A Metal device and its command queues; clones share them, and any thread may use them.
/// An operation runs after the operations committed before it that use its buffers; the others
/// may run concurrently on other queues and finish independently.
///
/// With `METAL_CAPTURE_ENABLED=1` and `METAL_OXIDE_CAPTURE=<file>.gputrace` set at process start,
/// the first device records a GPU trace into that file until its last clone drops, including the
/// clones its pipelines hold.
#[derive(Clone)]
pub struct Device(Arc<Context>);

struct Context {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
    backend: Backend,
    id: u64,
    failures: Arc<FailurePool>,
    snapshot: OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    log: OnceLock<Retained<ProtocolObject<dyn MTLLogState>>>,
    libraries: Libraries,
    capture: bool,
}

impl Drop for Context {
    fn drop(&mut self) {
        if self.capture {
            super::capture::stop();
        }
    }
}

impl Device {
    pub fn system_default() -> Result<Self> {
        let raw = MTLCreateSystemDefaultDevice().ok_or(Error::DeviceUnavailable)?;
        Self::from_native(raw)
    }

    /// Opens an independent context for each available device in native enumeration order.
    /// Returns an error if any device cannot be initialized. The order is not a persistent identity.
    pub fn all() -> Result<Vec<Self>> {
        MTLCopyAllDevices()
            .into_iter()
            .map(Self::from_native)
            .collect()
    }

    /// Retains a native device and creates new, independent command queues.
    /// Existing native queues and their pending work are not imported.
    pub fn from_native(raw: Retained<ProtocolObject<dyn MTLDevice>>) -> Result<Self> {
        // Artifacts target macOS 15, and source compilation sets its math options.
        if !objc2::available!(macos = 15.0) {
            return Err(Error::UnsupportedDevice(
                "metal-oxide requires macOS 15 or later".into(),
            ));
        }
        autoreleasepool(|_| {
            if !raw.hasUnifiedMemory() {
                return Err(Error::UnsupportedDevice(format!(
                    "{} does not have unified memory",
                    raw.name()
                )));
            }
            let backend = Backend::new(&raw)?;
            #[allow(deprecated, reason = "try_update is unavailable at the MSRV")]
            let id = NEXT_DEVICE_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .map_err(|_| Error::Command("device identity space exhausted".into()))?;
            let capture = super::capture::start(&raw)?;
            Ok(Self(Arc::new(Context {
                raw,
                backend,
                id,
                failures: Arc::default(),
                snapshot: OnceLock::new(),
                log: OnceLock::new(),
                libraries: Libraries::default(),
                capture,
            })))
        })
    }

    pub fn name(&self) -> String {
        autoreleasepool(|_| self.raw().name().to_string())
    }

    /// Borrows the native device for resource creation and external Metal integration.
    pub fn as_native(&self) -> &ProtocolObject<dyn MTLDevice> {
        self.raw()
    }

    /// The device's recommended working-set size in bytes, not currently free memory
    /// or a guarantee that an allocation of this size will succeed.
    pub fn recommended_max_working_set_size(&self) -> u64 {
        self.raw().recommendedMaxWorkingSetSize()
    }

    pub fn buffer_from_slice<T: GpuValue>(&self, values: &[T]) -> Result<Buffer<T>> {
        Buffer::from_slice(self, values)
    }

    pub fn buffer_zeroed<T: GpuValue>(&self, len: usize) -> Result<Buffer<T>> {
        Buffer::zeroed(self, len)
    }

    pub fn max_block_dimensions(&self) -> Dim3 {
        let limits = self.raw().maxThreadsPerThreadgroup();
        let axis = |value: usize| value.min(u32::MAX as usize) as u32;
        Dim3::new(axis(limits.width), axis(limits.height), axis(limits.depth))
    }

    /// Maximum combined static and dynamic threadgroup storage per block, in bytes.
    pub fn max_threadgroup_memory_per_block(&self) -> usize {
        self.raw().maxThreadgroupMemoryLength()
    }

    pub(super) fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.0.raw
    }

    pub(super) fn backend(&self) -> &Backend {
        &self.0.backend
    }

    pub(super) fn id(&self) -> u64 {
        self.0.id
    }

    pub(super) fn failure_pool(&self) -> &Arc<FailurePool> {
        &self.0.failures
    }

    /// The built-in kernel that copies a command buffer's failure word into a later gate.
    pub(super) fn snapshot(
        &self,
    ) -> Result<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        if let Some(pipeline) = self.0.snapshot.get() {
            return Ok(pipeline);
        }
        let module = Module::from_source(self, SNAPSHOT_SOURCE)?;
        let pipeline = super::pipeline::compile(self, &module, SNAPSHOT_KERNEL)?;
        Ok(self.0.snapshot.get_or_init(|| pipeline))
    }

    /// Returns an embedded kernel's cached pipeline without waiting for the cache lock.
    pub(super) fn cached_kernel(
        &self,
        library: &'static Library,
        entry: &'static str,
    ) -> Option<Pipeline> {
        self.0
            .libraries
            .cached(library, entry)
            .map(|compiled| Pipeline::from_compiled(self.clone(), compiled))
    }

    /// Loads and compiles an embedded kernel on a preparation worker.
    pub(super) fn kernel(
        &self,
        library: &'static Library,
        entry: &'static str,
    ) -> Result<Pipeline> {
        let compiled = self.0.libraries.pipeline(self, library, entry)?;
        Ok(Pipeline::from_compiled(self.clone(), compiled))
    }

    /// The log state that command buffers carry once a logging artifact is loaded.
    pub(super) fn log_state(&self) -> Option<&ProtocolObject<dyn MTLLogState>> {
        self.0.log.get().map(|state| &**state)
    }

    pub(super) fn enable_logging(&self) -> Result<()> {
        if self.0.log.get().is_none() {
            let state = super::logging::log_state(self.raw())?;
            // A concurrent loader may have installed its state first; either one serves.
            let _ = self.0.log.set(state);
        }
        Ok(())
    }
}
