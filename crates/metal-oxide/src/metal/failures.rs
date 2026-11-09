use std::{ops::Range, sync::Arc};

use metal_oxide_artifact::Check;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::sync::{Mutex, lock};
use crate::{Error, KernelFailure, Result};

const WORD_BYTES: usize = 4;
const CHUNK_WORDS: usize = 1024;
/// Never written: the gate of dispatches that no fallible dispatch precedes.
const ZERO: usize = 0;
/// The most words one snapshot dispatch reads, one per thread of its single block.
const SNAPSHOT_WIDTH: usize = 256;

/// Shared storage for gate, failure and length words.
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
        chunk.clear();
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

    fn clear(&self) {
        // SAFETY: no GPU work uses the chunk.
        unsafe {
            self.0
                .contents()
                .cast::<u8>()
                .write_bytes(0, CHUNK_WORDS * WORD_BYTES)
        };
    }
}

/// Zeroed chunks of one device, reused after their command buffer completes.
#[derive(Default)]
pub(super) struct FailurePool(Mutex<Vec<Chunk>>);

impl FailurePool {
    fn take(&self, device: &ProtocolObject<dyn MTLDevice>) -> Result<Chunk> {
        match lock(&self.0).pop() {
            Some(chunk) => Ok(chunk),
            None => Chunk::new(device),
        }
    }
}

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

/// Consecutive words of a command buffer's chunk that one snapshot dispatch reads.
pub(super) struct Segment {
    pub(super) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) offset: usize,
    pub(super) count: usize,
}

/// The gate of a level's generated dispatches.
pub(super) enum Gate {
    /// A word that no dispatch of the level writes.
    Word(Slot),
    /// A fresh word that snapshot dispatches over `sources` set before the level starts.
    Snapshot { gate: Slot, sources: Vec<Segment> },
}

/// Words handed out in order from chunks of the command buffer.
#[derive(Default)]
struct Region {
    chunks: Vec<usize>,
    used: usize,
}

impl Region {
    /// Reserves `count` words at a multiple of `align` inside one chunk.
    fn allocate(
        &mut self,
        chunks: &mut Vec<Chunk>,
        pool: &FailurePool,
        device: &ProtocolObject<dyn MTLDevice>,
        count: usize,
        align: usize,
    ) -> Result<usize> {
        let mut start = self.used.next_multiple_of(align);
        if start + count > self.chunks.len() * CHUNK_WORDS {
            start = self.chunks.len() * CHUNK_WORDS;
            let chunk = pool.take(device)?;
            self.chunks.push(chunks.len());
            chunks.push(chunk);
        }
        self.used = start + count;
        Ok(start)
    }

    fn slot(&self, index: usize) -> Slot {
        Slot {
            chunk: self.chunks[index / CHUNK_WORDS],
            offset: index % CHUNK_WORDS * WORD_BYTES,
        }
    }
}

struct Fallible {
    kernel: Arc<KernelChecks>,
    dispatch: usize,
    word: usize,
}

/// Why a command buffer's kernels did not all complete.
pub(super) enum Failed {
    /// One of its own dispatches failed a check.
    Kernel(KernelFailure),
    /// A command buffer whose writes it read failed first.
    Dependency(KernelFailure),
}

/// The gate, failure and length words of one command buffer.
///
/// Every fallible dispatch sets its own failure word. The dispatches between two barriers form
/// a level and share a gate that none of them writes: zero, a failure word of an earlier level,
/// or a snapshot of several such words taken before the level.
pub(super) struct Failures {
    pool: Arc<FailurePool>,
    chunks: Vec<Chunk>,
    words: Region,
    tables: Region,
    fallible: Vec<Fallible>,
    /// The latest snapshot gate, which covers every word before it.
    gate: Option<usize>,
    /// Words before this index were written by levels that have ended.
    closed: usize,
    propagation: Mutex<Propagation>,
}

/// What a command buffer passes on to the command buffers that read its writes.
enum Propagation {
    /// The command buffers whose writes it reads, kept until it knows its failure.
    Upstream(Vec<Arc<Failures>>),
    /// The failure behind it, recorded by its completion handler.
    Root(Option<KernelFailure>),
}

impl Failures {
    pub(super) fn new(pool: Arc<FailurePool>) -> Self {
        Self {
            pool,
            chunks: Vec::new(),
            words: Region::default(),
            tables: Region::default(),
            fallible: Vec::new(),
            gate: None,
            closed: 0,
            propagation: Mutex::new(Propagation::Upstream(Vec::new())),
        }
    }

    /// Gates every dispatch on the words of the command buffers in `upstream`, which this one
    /// reads from: before the first level, snapshots set a fresh gate when any of them is set.
    pub(super) fn follow(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        upstream: Vec<Arc<Failures>>,
    ) -> Result<Option<Gate>> {
        if upstream.is_empty() {
            return Ok(None);
        }
        let sources = upstream
            .iter()
            .flat_map(|failures| failures.segments(0..failures.words.used))
            .collect::<Vec<_>>();
        *lock(&self.propagation) = Propagation::Upstream(upstream);
        if sources.is_empty() {
            return Ok(None);
        }
        self.reserve_zero(device)?;
        let index = self.allocate_word(device)?;
        self.gate = Some(index);
        self.closed = self.words.used;
        Ok(Some(Gate::Snapshot {
            gate: self.words.slot(index),
            sources,
        }))
    }

    /// Whether a word besides zero exists that a later command buffer could gate on.
    pub(super) fn has_words(&self) -> bool {
        self.words.used > ZERO + 1
    }

    /// Ends the current level: its failure words become sources of later gates.
    pub(super) fn close_level(&mut self) {
        self.closed = self.words.used;
    }

    /// The words a gate must cover: the latest snapshot gate, or zero, and the failure words of
    /// ended levels after it.
    fn sources(&self) -> Range<usize> {
        self.gate.unwrap_or(ZERO)..self.closed.max(ZERO + 1)
    }

    /// Whether the next level needs a snapshot because several words may be set.
    fn needs_snapshot(&self) -> bool {
        self.sources().len() > if self.gate.is_some() { 1 } else { 2 }
    }

    /// The gate of a level that has no dispatch encoded yet.
    pub(super) fn gate(&mut self, device: &ProtocolObject<dyn MTLDevice>) -> Result<Gate> {
        self.reserve_zero(device)?;
        let sources = self.sources();
        if !self.needs_snapshot() {
            return Ok(Gate::Word(self.words.slot(sources.end - 1)));
        }
        let index = self.allocate_word(device)?;
        self.gate = Some(index);
        self.closed = self.words.used;
        Ok(Gate::Snapshot {
            gate: self.words.slot(index),
            sources: self.segments(sources),
        })
    }

    /// The words in `range`, split by chunk and snapshot width.
    fn segments(&self, range: Range<usize>) -> Vec<Segment> {
        let mut segments = Vec::new();
        let mut start = range.start;
        while start < range.end {
            let chunk_end = (start / CHUNK_WORDS + 1) * CHUNK_WORDS;
            let end = range.end.min(chunk_end).min(start + SNAPSHOT_WIDTH);
            let slot = self.words.slot(start);
            segments.push(Segment {
                buffer: self.chunks[slot.chunk].0.clone(),
                offset: slot.offset,
                count: end - start,
            });
            start = end;
        }
        segments
    }

    /// The word that the failing threads of a fallible dispatch set.
    pub(super) fn failure_word(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        kernel: &Arc<KernelChecks>,
        dispatch: usize,
    ) -> Result<Slot> {
        self.reserve_zero(device)?;
        let word = self.allocate_word(device)?;
        self.fallible.push(Fallible {
            kernel: Arc::clone(kernel),
            dispatch,
            word,
        });
        Ok(self.words.slot(word))
    }

    /// Writes `values` into aligned words that one dispatch reads as a `ulong` table.
    pub(super) fn table(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        values: &[u64],
    ) -> Result<Slot> {
        let index =
            self.tables
                .allocate(&mut self.chunks, &self.pool, device, values.len() * 2, 2)?;
        let slot = self.tables.slot(index);
        self.chunks[slot.chunk].write(slot.offset / WORD_BYTES, values);
        Ok(slot)
    }

    pub(super) fn buffer(&self, slot: Slot) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.chunks[slot.chunk].0
    }

    /// Why the command buffer failed, read once by its completion handler. Records the root
    /// failure for later command buffers and releases the upstream ones.
    pub(super) fn finish(&self) -> Option<Failed> {
        let mut propagation = lock(&self.propagation);
        let failed = match self.failure() {
            Some(failure) => Some(Failed::Kernel(failure)),
            None => propagation.upstream_root().map(Failed::Dependency),
        };
        let root = failed
            .as_ref()
            .map(|(Failed::Kernel(failure) | Failed::Dependency(failure))| failure.clone());
        *propagation = Propagation::Root(root);
        failed
    }

    /// The failure that a command buffer reading from this one reports, after this one's GPU
    /// work has completed.
    fn root(&self) -> Option<KernelFailure> {
        let propagation = lock(&self.propagation);
        match &*propagation {
            Propagation::Root(root) => root.clone(),
            Propagation::Upstream(_) => self.failure().or_else(|| propagation.upstream_root()),
        }
    }

    /// The failed dispatch pushed first, read after the command buffer completes.
    fn failure(&self) -> Option<KernelFailure> {
        self.fallible
            .iter()
            .filter_map(|fallible| {
                let slot = self.words.slot(fallible.word);
                let state = self.chunks[slot.chunk].word(slot.offset / WORD_BYTES);
                (state != 0).then_some((fallible, state))
            })
            .min_by_key(|(fallible, _)| fallible.dispatch)
            .map(|(fallible, state)| fallible.report(state.wrapping_sub(2)))
    }

    fn reserve_zero(&mut self, device: &ProtocolObject<dyn MTLDevice>) -> Result<()> {
        if self.words.used == 0 {
            self.allocate_word(device)?;
            self.closed = self.words.used;
        }
        Ok(())
    }

    fn allocate_word(&mut self, device: &ProtocolObject<dyn MTLDevice>) -> Result<usize> {
        self.words
            .allocate(&mut self.chunks, &self.pool, device, 1, 1)
    }
}

impl Drop for Failures {
    /// Runs after completion or for an uncommitted batch, so no GPU work uses the chunks.
    fn drop(&mut self) {
        for chunk in &self.chunks {
            chunk.clear();
        }
        lock(&self.pool.0).append(&mut self.chunks);
    }
}

impl Propagation {
    /// The failure behind the first upstream command buffer that has one.
    fn upstream_root(&self) -> Option<KernelFailure> {
        match self {
            Self::Upstream(upstream) => upstream.iter().find_map(|failures| failures.root()),
            Self::Root(_) => None,
        }
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

#[cfg(all(test, loom))]
mod loom;
