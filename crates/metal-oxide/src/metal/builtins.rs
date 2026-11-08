use std::sync::{Arc, OnceLock};

use super::{Device, Module, pipeline::Compiled};
use crate::Result;

/// The runtime's kernels; sizes are a `uint4` of whole 16-byte vectors, the bytes after them and
/// the pattern's period in vectors. Byte pointers let API validation accept buffers under 16 B.
const SOURCE: &str = "#include <metal_stdlib>
using namespace metal;

kernel void metal_oxide_snapshot(const device uint* words [[buffer(0)]], device atomic_uint* gate [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (words[i] != 0u) {
        atomic_store_explicit(gate, 1u, memory_order_relaxed);
    }
}

kernel void metal_oxide_copy(const device uchar* source [[buffer(0)]], device uchar* destination [[buffer(1)]], constant uint4& size [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < size.x) {
        ((device uint4*)destination)[i] = ((const device uint4*)source)[i];
    } else if (i == size.x) {
        for (uint b = size.x * 16; b < size.x * 16 + size.y; ++b) {
            destination[b] = source[b];
        }
    }
}

kernel void metal_oxide_fill(device uchar* bytes [[buffer(0)]], constant uint4& pattern [[buffer(1)]], constant uint4& size [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    device uint4* destination = (device uint4*)bytes;
    uint first = i * 4;
    if (first + 4 <= size.x) {
        destination[first] = pattern;
        destination[first + 1] = pattern;
        destination[first + 2] = pattern;
        destination[first + 3] = pattern;
    } else if (first <= size.x) {
        for (uint v = first; v < size.x; ++v) {
            destination[v] = pattern;
        }
        constant uchar* tail = (constant uchar*)&pattern;
        for (uint b = 0; b < size.y; ++b) {
            bytes[size.x * 16 + b] = tail[b];
        }
    }
}

kernel void metal_oxide_fill_period(device uchar* bytes [[buffer(0)]], constant uint4* pattern [[buffer(1)]], constant uint4& size [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < size.x) {
        ((device uint4*)bytes)[i] = pattern[i % size.z];
    } else if (i == size.x) {
        constant uchar* tail = (constant uchar*)(pattern + i % size.z);
        for (uint b = 0; b < size.y; ++b) {
            bytes[size.x * 16 + b] = tail[b];
        }
    }
}
";

/// A kernel that a device context compiles from the runtime's source on first use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Builtin {
    /// Sets a fresh gate when any failure word it reads is set.
    Snapshot,
    /// Copies 16 bytes per thread.
    Copy,
    /// Writes a 16-byte pattern, 64 bytes per thread.
    Fill,
    /// Writes a pattern of several 16-byte vectors, 16 bytes per thread.
    FillPeriod,
}

impl Builtin {
    const ALL: [Self; 4] = [Self::Snapshot, Self::Copy, Self::Fill, Self::FillPeriod];

    pub(super) fn entry(self) -> &'static str {
        match self {
            Self::Snapshot => "metal_oxide_snapshot",
            Self::Copy => "metal_oxide_copy",
            Self::Fill => "metal_oxide_fill",
            Self::FillPeriod => "metal_oxide_fill_period",
        }
    }
}

/// The built-in library of a device context and its pipelines; entries hold no `Device`.
#[derive(Default)]
pub(super) struct Builtins {
    module: OnceLock<Module>,
    pipelines: [OnceLock<Arc<Compiled>>; Builtin::ALL.len()],
}

impl Builtins {
    pub(super) fn cached(&self, kernel: Builtin) -> Option<&Arc<Compiled>> {
        self.pipelines[kernel as usize].get()
    }

    /// The pipeline of `kernel`; concurrent first uses may compile it twice and keep one.
    pub(super) fn compile(&self, device: &Device, kernel: Builtin) -> Result<&Arc<Compiled>> {
        if let Some(compiled) = self.cached(kernel) {
            return Ok(compiled);
        }
        let module = match self.module.get() {
            Some(module) => module,
            None => {
                let module = Module::from_source(device, SOURCE)?;
                self.module.get_or_init(|| module)
            }
        };
        let compiled = Arc::new(Compiled::new(device, module, kernel.entry())?);
        Ok(self.pipelines[kernel as usize].get_or_init(|| compiled))
    }
}
