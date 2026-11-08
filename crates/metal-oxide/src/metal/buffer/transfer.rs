use super::Buffer;
use crate::{
    DynamicLaunchConfig, Error, GpuValue, Result,
    metal::{Argument, Launch, builtins::Builtin},
};

/// Threads per block of the runtime's copy and fill dispatches.
const BLOCK: u32 = 256;

impl<T: GpuValue> Buffer<T> {
    /// Copies `source`, which has the same length, into this buffer on the GPU when the returned
    /// operation runs; it follows earlier work on either buffer like any launch.
    pub fn copy_from<'a>(
        &'a mut self,
        source: &'a Buffer<T>,
    ) -> Launch<(&'a Buffer<T>, &'a mut Buffer<T>), 3> {
        let size = vector_size(self.len * T::SIZE);
        let threads = size
            .as_ref()
            .map_or(0, |[vectors, tail, ..]| vectors + u32::from(*tail > 0));
        let check = if source.len == self.len {
            size.map(drop)
        } else {
            Err(Error::InvalidLaunch(format!(
                "copying {} elements into a buffer of {}",
                source.len, self.len
            )))
        };
        let device = self.device.clone();
        // SAFETY: each thread copies its own 16 bytes of both buffers, the last one the tail.
        unsafe {
            Launch::builtin(
                &device,
                Builtin::Copy,
                check,
                config(threads),
                (source, self),
                |(source, destination)| {
                    let size = vector_size(destination.len() * T::SIZE)?;
                    Ok([
                        Argument::read(*source),
                        Argument::write(destination),
                        Argument::vectors(bytes(size)),
                    ])
                },
            )
        }
    }

    /// Sets every element to `value` on the GPU when the returned operation runs; it follows
    /// earlier work on this buffer like any launch.
    pub fn fill(&mut self, value: T) -> Launch<(&mut Buffer<T>, T), 3> {
        let size = vector_size(self.len * T::SIZE);
        let (kernel, threads) = match size {
            Ok([0, 0, ..]) | Err(_) => (Builtin::Fill, 0),
            Ok([vectors, ..]) if pattern(value).len() == 16 => (Builtin::Fill, vectors / 4 + 1),
            Ok([vectors, tail, ..]) => (Builtin::FillPeriod, vectors + u32::from(tail > 0)),
        };
        let device = self.device.clone();
        // SAFETY: each thread writes its own 64 or 16 bytes of the buffer, the last one the tail.
        unsafe {
            Launch::builtin(
                &device,
                kernel,
                size.map(drop),
                config(threads),
                (self, value),
                |(buffer, value)| {
                    let pattern = pattern(*value);
                    let mut size = vector_size(buffer.len() * T::SIZE)?;
                    size[2] = (pattern.len() / 16) as u32;
                    Ok([
                        Argument::write(buffer),
                        Argument::vectors(pattern),
                        Argument::vectors(bytes(size)),
                    ])
                },
            )
        }
    }
}

/// Whole 16-byte vectors, the bytes after them and a pattern period of one vector.
fn vector_size(bytes: usize) -> Result<[u32; 4]> {
    // Fill threads index four vectors each, so four more must not overflow `uint`.
    let vectors = u32::try_from(bytes / 16)
        .ok()
        .filter(|vectors| *vectors <= u32::MAX - 4)
        .ok_or_else(|| Error::InvalidLaunch(format!("{bytes} B exceed one copy or fill")))?;
    Ok([vectors, (bytes % 16) as u32, 1, 0])
}

/// `value`'s bytes repeated to the shortest whole number of 16-byte vectors.
fn pattern<T: GpuValue>(value: T) -> Vec<u8> {
    let mut element = vec![0; T::SIZE];
    value.encode(&mut element);
    let uniform = element.windows(2).all(|pair| pair[0] == pair[1]);
    let length = if uniform || 16 % T::SIZE == 0 {
        16
    } else {
        T::SIZE / gcd(T::SIZE, 16) * 16
    };
    element.iter().copied().cycle().take(length).collect()
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn bytes(words: [u32; 4]) -> Vec<u8> {
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn config(threads: u32) -> DynamicLaunchConfig {
    DynamicLaunchConfig::new(
        crate::Dim3::x(threads.div_ceil(BLOCK)),
        crate::Dim3::x(BLOCK),
    )
}

#[cfg(test)]
mod tests;
