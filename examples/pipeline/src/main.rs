/// Two stages whose kernels share an affine helper and a generic scaling body.
#[metal_oxide::kernels]
pub mod stages {
    use metal_oxide::device::{ReadBuffer, WriteBuffer, block_dim, block_idx, thread_idx};

    struct Affine {
        scale: f32,
        offset: f32,
    }

    impl Affine {
        fn apply(self, value: f32) -> f32 {
            value * self.scale + self.offset
        }
    }

    unsafe fn scale<const BLOCK: u32, const SCALE: u32>(
        input: ReadBuffer<f32>,
        output: WriteBuffer<f32>,
        n: u32,
    ) {
        let i = block_idx().x * BLOCK + thread_idx().x;
        if i < n {
            let affine = Affine {
                scale: SCALE as f32,
                offset: 3.0,
            };
            // SAFETY: entrypoint callers supply n elements and one writer per index.
            unsafe { output.store_unchecked(i, affine.apply(input.load_unchecked(i))) };
        }
    }

    /// # Safety
    /// Buffers cover n elements, do not overlap, and the launch uses 128 threads per block.
    #[kernel(block = (128, 1, 1))]
    pub unsafe fn scale_128(input: ReadBuffer<f32>, output: WriteBuffer<f32>, n: u32) {
        // SAFETY: the entrypoint contract fixes the index calculation and buffer bounds.
        unsafe { scale::<128, 2>(input, output, n) };
    }

    /// # Safety
    /// Buffers cover n elements, do not overlap, and the launch uses 256 threads per block.
    #[kernel(block = (256, 1, 1))]
    pub unsafe fn scale_256(input: ReadBuffer<f32>, output: WriteBuffer<f32>, n: u32) {
        // SAFETY: the entrypoint contract fixes the index calculation and buffer bounds.
        unsafe { scale::<256, 4>(input, output, n) };
    }

    /// # Safety
    /// Buffers cover n elements, do not overlap, and the launch is one dimensional.
    #[kernel]
    pub unsafe fn finish(input: ReadBuffer<f32>, output: WriteBuffer<f32>, n: u32) {
        let i = block_idx().x * block_dim().x + thread_idx().x;
        if i < n {
            let affine = Affine {
                scale: 5.0,
                offset: 7.0,
            };
            // SAFETY: each active thread reads an initialized input and writes its own output.
            unsafe { output.store_unchecked(i, affine.apply(input.load_unchecked(i))) };
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod run;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    run::block_on(run::verify())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("pipeline runs on macOS with Apple Silicon");
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests;
