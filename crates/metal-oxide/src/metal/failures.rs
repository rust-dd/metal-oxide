use std::sync::Arc;

use metal_oxide_artifact::Check;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::sync::{Mutex, lock};
use crate::{Error, KernelFailure, Result};

const WORD_BYTES: usize = 4;
const CHUNK_WORDS: usize = 1024;
/// Never written: the gate of dispatches that no fallible dispatch precedes.
const ZERO: usize = 0;
/// Failing threads store `check + 2` here.
const FAILURE: usize = 1;

pub(super) const SNAPSHOT_KERNEL: &str = "metal_oxide_snapshot";

/// Copies the failure word into a later fallible dispatch's gate, which must not be a word
/// that the dispatch's own failing threads write.
pub(super) const SNAPSHOT_SOURCE: &str = "#include <metal_stdlib>
using namespace metal;

kernel void metal_oxide_snapshot(const device uint* failure [[buffer(0)]], device uint* gate [[buffer(1)]]) {
    *gate = *failure;
}
";

/// Shared storage for gate and failure words.
struct Chunk(Retained<ProtocolObject<dyn MTLBuffer>>);

// SAFETY: the runtime retains, releases and reads immutable properties of the buffer on any
// thread. Its bytes are written by the GPU before completion and read or cleared only after it.
unsafe impl Send for Chunk {}
// SAFETY: see the `Send` implementation; shared access never mutates the native object.
unsafe impl Sync for Chunk {}

impl Chunk {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let bytes = WORD_BYTES * CHUNK_WORDS;
        let raw = device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::AllocationFailed { bytes })?;
        let chunk = Self(raw);
        chunk.clear(CHUNK_WORDS);
        Ok(chunk)
    }

    fn write(&self, index: usize, values: &[u64]) {
        // SAFETY: the words are 8-byte aligned, inside the chunk, and its command buffer has
        // not been committed.
        unsafe {
            let words = self.0.contents().cast::<u32>().add(index).cast::<u64>();
            words
                .as_ptr()
                .copy_from_nonoverlapping(values.as_ptr(), values.len());
        }
    }

    fn word(&self, index: usize) -> u32 {
        // SAFETY: the index is inside the chunk, and its command buffer has completed.
        unsafe { self.0.contents().cast::<u32>().add(index).read() }
    }

    fn clear(&self, words: usize) {
        // SAFETY: no GPU work uses the chunk, and `words` fits inside it.
        unsafe {
            self.0
                .contents()
                .cast::<u8>()
                .write_bytes(0, words * WORD_BYTES)
        };
    }
}

/// Zeroed chunks of one device, reused after their command buffer completes.
#[derive(Default)]
pub(super) struct FailurePool(Mutex<Vec<Chunk>>);

/// The generated kernel behind a pipeline: its name, hidden bindings and check table.
pub(super) struct KernelChecks {
    pub(super) name: String,
    pub(super) gate_binding: u32,
    pub(super) fallible: bool,
    pub(super) lengths: bool,
    pub(super) checks: Arc<[Check]>,
}

/// A word inside the chunks of one command buffer.
#[derive(Clone, Copy)]
pub(super) struct Slot {
    chunk: usize,
    pub(super) offset: usize,
}

/// The words a generated dispatch binds after its parameters.
pub(super) struct Assignment {
    pub(super) gate: Slot,
    pub(super) failure: Option<Slot>,
    /// The failure word must be copied into the gate before the dispatch.
    pub(super) snapshot: bool,
}

struct Fallible {
    kernel: Arc<KernelChecks>,
    dispatch: usize,
    gate: Slot,
}

/// The gate and failure words of one command buffer. Once a dispatch fails, later infallible
/// dispatches gate on the failure word and later fallible ones on a snapshot taken before them.
pub(super) struct Failures {
    pool: Arc<FailurePool>,
    chunks: Vec<Chunk>,
    used: usize,
    fallible: Vec<Fallible>,
}

impl Failures {
    pub(super) fn new(pool: Arc<FailurePool>) -> Self {
        Self {
            pool,
            chunks: Vec::new(),
            used: 0,
            fallible: Vec::new(),
        }
    }

    pub(super) fn assign(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        fallible: bool,
    ) -> Result<Assignment> {
        if self.used == 0 {
            self.reserve(device)?;
            self.used = FAILURE + 1;
        }
        let first = self.fallible.is_empty();
        Ok(match (fallible, first) {
            (false, _) => Assignment {
                gate: Self::slot(if first { ZERO } else { FAILURE }),
                failure: None,
                snapshot: false,
            },
            (true, true) => Assignment {
                gate: Self::slot(ZERO),
                failure: Some(Self::slot(FAILURE)),
                snapshot: false,
            },
            (true, false) => Assignment {
                gate: self.allocate(device)?,
                failure: Some(Self::slot(FAILURE)),
                snapshot: true,
            },
        })
    }

    /// Records an encoded fallible dispatch and the gate it reads.
    pub(super) fn record(&mut self, kernel: &Arc<KernelChecks>, dispatch: usize, gate: Slot) {
        self.fallible.push(Fallible {
            kernel: Arc::clone(kernel),
            dispatch,
            gate,
        });
    }

    pub(super) fn failure_slot(&self) -> Slot {
        Self::slot(FAILURE)
    }

    pub(super) fn buffer(&self, slot: Slot) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.chunks[slot.chunk].0
    }

    /// The failed dispatch, read after the command buffer completes.
    pub(super) fn failure(&self) -> Option<KernelFailure> {
        if self.fallible.is_empty() {
            return None;
        }
        let state = self.read(Self::slot(FAILURE));
        if state == 0 {
            return None;
        }
        // Gates after a failure hold its snapshot, so the failed dispatch is the last whose
        // gate stayed zero.
        self.fallible
            .iter()
            .rev()
            .find(|fallible| self.read(fallible.gate) == 0)
            .map(|failed| failed.report(state.wrapping_sub(2)))
    }

    fn read(&self, slot: Slot) -> u32 {
        self.chunks[slot.chunk].word(slot.offset / WORD_BYTES)
    }

    /// Writes `values` into aligned words that one dispatch reads as a `ulong` table.
    pub(super) fn table(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        values: &[u64],
    ) -> Result<Slot> {
        let words = values.len() * 2;
        let mut start = self.used.next_multiple_of(2);
        if start + words > self.chunks.len() * CHUNK_WORDS {
            start = self.chunks.len() * CHUNK_WORDS;
            self.reserve(device)?;
        }
        self.used = start + words;
        let slot = Self::slot(start);
        self.chunks[slot.chunk].write(slot.offset / WORD_BYTES, values);
        Ok(slot)
    }

    fn allocate(&mut self, device: &ProtocolObject<dyn MTLDevice>) -> Result<Slot> {
        if self.used == self.chunks.len() * CHUNK_WORDS {
            self.reserve(device)?;
        }
        self.used += 1;
        Ok(Self::slot(self.used - 1))
    }

    fn reserve(&mut self, device: &ProtocolObject<dyn MTLDevice>) -> Result<()> {
        let reused = lock(&self.pool.0).pop();
        self.chunks.push(match reused {
            Some(chunk) => chunk,
            None => Chunk::new(device)?,
        });
        Ok(())
    }

    fn slot(index: usize) -> Slot {
        Slot {
            chunk: index / CHUNK_WORDS,
            offset: index % CHUNK_WORDS * WORD_BYTES,
        }
    }
}

impl Drop for Failures {
    /// Runs after completion or for an uncommitted batch, so no GPU work uses the chunks.
    fn drop(&mut self) {
        let mut remaining = self.used;
        for chunk in &self.chunks {
            chunk.clear(remaining.min(CHUNK_WORDS));
            remaining = remaining.saturating_sub(CHUNK_WORDS);
        }
        lock(&self.pool.0).append(&mut self.chunks);
    }
}

impl Fallible {
    fn report(&self, check: u32) -> KernelFailure {
        let (message, file, line, column) = match self.kernel.checks.get(check as usize) {
            Some(check) => (
                check.kind.message(),
                check.location.file.clone(),
                check.location.line,
                check.location.column,
            ),
            None => (
                format!("unknown runtime check {check}"),
                String::new(),
                0,
                0,
            ),
        };
        KernelFailure {
            kernel: self.kernel.name.clone(),
            dispatch: self.dispatch,
            message,
            file,
            line,
            column,
        }
    }
}
