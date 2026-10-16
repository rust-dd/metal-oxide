use std::{
    collections::HashMap,
    sync::{Arc, Mutex, PoisonError, TryLockError},
};

use crate::{Error, Result};

use super::{Device, Module, pipeline::Compiled};

/// The compiled kernels that a `#[kernel]` function or a `#[kernels]` module embeds.
#[doc(hidden)]
pub struct Library {
    manifest: &'static str,
    metallib: &'static [u8],
    abi: &'static str,
}

impl Library {
    /// An artifact's manifest and library with the ABI hash its launch functions were
    /// generated for.
    pub const fn new(manifest: &'static str, metallib: &'static [u8], abi: &'static str) -> Self {
        Self {
            manifest,
            metallib,
            abi,
        }
    }

    /// The library as a module of `device`, for pipelines that code encodes itself.
    pub fn module(&self, device: &Device) -> Result<Module> {
        if self.metallib.is_empty() {
            return Err(Error::Library(
                "the kernels were not built for this target".into(),
            ));
        }
        let module = Module::from_bytes(device, self.manifest, self.metallib)?;
        module.verify_abi(self.abi)?;
        Ok(module)
    }
}

/// The embedded libraries a device has loaded and their pipelines, created on first launch.
/// Entries hold no `Device`, so the device context can own them.
#[derive(Default)]
pub(super) struct Libraries(Mutex<HashMap<usize, Loaded>>);

struct Loaded {
    module: Arc<Module>,
    pipelines: HashMap<&'static str, Arc<Compiled>>,
}

impl Libraries {
    pub(super) fn cached(
        &self,
        library: &'static Library,
        entry: &'static str,
    ) -> Option<Arc<Compiled>> {
        let loaded = match self.0.try_lock() {
            Ok(loaded) => loaded,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return None,
        };
        let key = std::ptr::from_ref(library).addr();
        loaded.get(&key)?.pipelines.get(entry).cloned()
    }

    pub(super) fn pipeline(
        &self,
        device: &Device,
        library: &'static Library,
        entry: &'static str,
    ) -> Result<Arc<Compiled>> {
        let key = std::ptr::from_ref(library).addr();
        let cached = self.lock().get(&key).map(|loaded| {
            (
                loaded.pipelines.get(entry).cloned(),
                Arc::clone(&loaded.module),
            )
        });
        let module = match cached {
            Some((Some(compiled), _)) => return Ok(compiled),
            Some((None, module)) => module,
            None => {
                let module = library.module(device)?;
                let mut loaded = self.lock();
                let entry = loaded.entry(key).or_insert_with(|| Loaded {
                    module: Arc::new(module),
                    pipelines: HashMap::new(),
                });
                Arc::clone(&entry.module)
            }
        };
        let compiled = Arc::new(Compiled::new(device, &module, entry)?);
        let mut loaded = self.lock();
        let pipelines = &mut loaded.get_mut(&key).expect("loaded above").pipelines;
        Ok(Arc::clone(pipelines.entry(entry).or_insert(compiled)))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<usize, Loaded>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
