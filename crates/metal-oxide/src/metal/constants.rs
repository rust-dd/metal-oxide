use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::argument::Binding;
use crate::{Error, Result};

pub(super) struct Constants {
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    offsets: [usize; 31],
}

impl Constants {
    /// Packs the value bytes of `bindings`, taken from `store`, into one shared buffer.
    ///
    /// # Safety
    ///
    /// Reused storage must come from `allocate` on `device`, with no remaining
    /// CPU accesses or encoded/pending GPU accesses.
    pub unsafe fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        bindings: &[Binding],
        store: &[u8],
        reusable: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    ) -> Result<Option<Self>> {
        let mut offsets = [0; 31];
        let mut length = 0_usize;
        let slots = bindings
            .iter()
            .filter(|binding| !matches!(binding, Binding::Threadgroup(_)));
        for (slot, binding) in slots.clone().enumerate() {
            if let Binding::Value { bytes, alignment } = binding {
                let alignment = (*alignment).max(4);
                length = length
                    .checked_add(alignment - 1)
                    .ok_or(Error::LengthOverflow)?
                    & !(alignment - 1);
                offsets[slot] = length;
                length = length
                    .checked_add(bytes.len())
                    .ok_or(Error::LengthOverflow)?;
            }
        }
        if length == 0 {
            return Ok(None);
        }
        let buffer = match reusable.filter(|buffer| buffer.length() >= length) {
            Some(buffer) => buffer,
            None => allocate(device, length)?,
        };
        for (slot, binding) in slots.enumerate() {
            if let Binding::Value { bytes, .. } = binding {
                // SAFETY: checked, aligned suballocations are disjoint and within
                // initialized, idle shared storage, before this command buffer commits.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        store[bytes.clone()].as_ptr(),
                        buffer.contents().cast::<u8>().as_ptr().add(offsets[slot]),
                        bytes.len(),
                    );
                }
            }
        }
        Ok(Some(Self { buffer, offsets }))
    }

    pub fn address(&self, slot: usize) -> u64 {
        self.buffer.gpuAddress() + self.offsets[slot] as u64
    }
}

pub(super) fn allocate(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let length = bytes.max(16);
    let maximum = device.maxBufferLength();
    if length > maximum {
        return Err(Error::BufferTooLarge {
            bytes: length,
            maximum,
        });
    }
    let buffer = device
        .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        .ok_or(Error::AllocationFailed { bytes: length })?;
    // SAFETY: fresh shared storage owns length writable bytes and is not committed.
    unsafe {
        buffer
            .contents()
            .cast::<u8>()
            .as_ptr()
            .write_bytes(0, length)
    };
    Ok(buffer)
}
