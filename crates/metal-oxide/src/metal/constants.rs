use metal_oxide_artifact::ParameterType;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::Argument;
use crate::{Error, Result};

pub(super) struct Constants {
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    offsets: [usize; 31],
}

impl Constants {
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        arguments: &[Argument<'_>],
    ) -> Result<Option<Self>> {
        let mut offsets = [0; 31];
        let mut length = 0_usize;
        let bindings = arguments.iter().filter(|a| a.threadgroup_bytes().is_none());
        for (slot, argument) in bindings.clone().enumerate() {
            if let ParameterType::Value { layout } = argument.ty() {
                let alignment = usize::try_from(layout.alignment)
                    .map_err(|_| Error::LengthOverflow)?
                    .max(4);
                length = length
                    .checked_add(alignment - 1)
                    .ok_or(Error::LengthOverflow)?
                    & !(alignment - 1);
                offsets[slot] = length;
                length = length
                    .checked_add(argument.value_bytes().unwrap().len())
                    .ok_or(Error::LengthOverflow)?;
            }
        }
        if length == 0 {
            return Ok(None);
        }
        let buffer = allocate(device, length)?;
        for (slot, argument) in bindings.enumerate() {
            if let Some(bytes) = argument.value_bytes() {
                // SAFETY: checked, aligned suballocations are disjoint and within
                // freshly initialized shared storage, before any GPU submission.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
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
    // SAFETY: fresh shared storage owns length writable bytes and is not submitted.
    unsafe {
        buffer
            .contents()
            .cast::<u8>()
            .as_ptr()
            .write_bytes(0, length)
    };
    Ok(buffer)
}
