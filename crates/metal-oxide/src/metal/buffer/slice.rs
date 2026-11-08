use std::{
    fmt,
    ops::{Bound, RangeBounds},
};

use super::Buffer;
use crate::GpuValue;

/// Elements of a buffer that a launch reads, such as `buffer.slice(..n)`; the kernel sees only
/// them, `requires` and `len()` included, and dependency tracking covers the whole buffer.
pub struct BufferSlice<'a, T: GpuValue> {
    pub(in crate::metal) buffer: &'a Buffer<T>,
    pub(in crate::metal) start: usize,
    pub(in crate::metal) len: usize,
}

/// Elements of a buffer that a launch writes, such as `buffer.slice_mut(..n)`, like
/// [`BufferSlice`].
pub struct BufferSliceMut<'a, T: GpuValue> {
    pub(in crate::metal) buffer: &'a mut Buffer<T>,
    pub(in crate::metal) start: usize,
    pub(in crate::metal) len: usize,
}

impl<T: GpuValue> fmt::Debug for BufferSlice<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferSlice")
            .field("buffer", self.buffer)
            .field("start", &self.start)
            .field("len", &self.len)
            .finish()
    }
}

impl<T: GpuValue> fmt::Debug for BufferSliceMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferSliceMut")
            .field("buffer", &*self.buffer)
            .field("start", &self.start)
            .field("len", &self.len)
            .finish()
    }
}

impl<T: GpuValue> Clone for BufferSlice<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: GpuValue> Copy for BufferSlice<'_, T> {}

impl<T: GpuValue> Buffer<T> {
    /// The elements in `range`; panics like slice indexing when it lies outside the buffer.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> BufferSlice<'_, T> {
        let (start, len) = elements(&range, self.len);
        BufferSlice {
            buffer: self,
            start,
            len,
        }
    }

    /// The elements in `range` for writing; panics like slice indexing outside the buffer.
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> BufferSliceMut<'_, T> {
        let (start, len) = elements(&range, self.len);
        BufferSliceMut {
            buffer: self,
            start,
            len,
        }
    }
}

impl<T: GpuValue> BufferSlice<'_, T> {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the slice has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: GpuValue> BufferSliceMut<'_, T> {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the slice has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The first element and count of `range` in `len` elements.
fn elements(range: &impl RangeBounds<usize>, len: usize) -> (usize, usize) {
    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start.checked_add(1).expect("range start overflows usize"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&end) => end.checked_add(1).expect("range end overflows usize"),
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };
    assert!(
        start <= end,
        "slice index starts at {start} but ends at {end}"
    );
    assert!(
        end <= len,
        "range end index {end} out of range for a buffer of length {len}"
    );
    (start, end - start)
}

#[cfg(test)]
mod tests;
