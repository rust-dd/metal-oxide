/// A read-only device buffer handle supplied by the kernel launch bindings.
#[repr(transparent)]
#[derive(Clone, Copy)]
#[cfg_attr(
    target_env = "metal",
    rustc_diagnostic_item = "metal_oxide_read_buffer"
)]
pub struct ReadBuffer<T> {
    ptr: *const T,
}

impl<T: Copy> ReadBuffer<T> {
    /// Reads an element without checking its index.
    ///
    /// # Safety
    ///
    /// The index must address an initialized element in this buffer. The read
    /// must not race with a write to that element.
    #[cfg_attr(
        target_env = "metal",
        rustc_diagnostic_item = "metal_oxide_buffer_load"
    )]
    pub unsafe fn load_unchecked(self, index: u32) -> T {
        // SAFETY: the caller guarantees bounds, initialization, and access discipline.
        unsafe { self.ptr.add(index as usize).read() }
    }
}

/// A writable device buffer handle supplied by the kernel launch bindings.
#[repr(transparent)]
#[derive(Clone, Copy)]
#[cfg_attr(
    target_env = "metal",
    rustc_diagnostic_item = "metal_oxide_write_buffer"
)]
pub struct WriteBuffer<T> {
    ptr: *mut T,
}

impl<T: Copy> WriteBuffer<T> {
    /// Writes an element without checking its index.
    ///
    /// # Safety
    ///
    /// The index must address an element in this buffer. The write must not race
    /// with any other access to that element.
    #[cfg_attr(
        target_env = "metal",
        rustc_diagnostic_item = "metal_oxide_buffer_store"
    )]
    pub unsafe fn store_unchecked(self, index: u32, value: T) {
        // SAFETY: the caller guarantees bounds and exclusive access to the element.
        unsafe { self.ptr.add(index as usize).write(value) }
    }
}

#[cfg(test)]
mod tests {
    use super::{ReadBuffer, WriteBuffer};

    #[test]
    fn raw_handles_access_only_the_selected_element() {
        let input = [2.0_f32, 3.0, 5.0];
        let read = ReadBuffer {
            ptr: input.as_ptr(),
        };
        let mut output = [-1.0_f32; 3];
        let write = WriteBuffer {
            ptr: output.as_mut_ptr(),
        };
        // SAFETY: both arrays are live; index 1 is valid and output has no concurrent access.
        unsafe { write.store_unchecked(1, read.load_unchecked(1) + 4.0) };
        assert_eq!(output, [-1.0, 7.0, -1.0]);
    }
}
