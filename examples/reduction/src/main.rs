use metal_oxide::device::{
    ReadBuffer, WriteBuffer, block_idx, thread_idx, thread_index, threadgroup,
};

/// Sums each block's 256 inputs into the block's output element; inputs past the end count as
/// zero, and blocks past the end of `output` write nothing.
///
/// # Safety
/// Nothing writes `input` while the kernel runs.
#[metal_oxide::kernel(block = (256, 1, 1))]
pub unsafe fn reduce(input: ReadBuffer<f32>, output: WriteBuffer<f32>) {
    let shared = threadgroup::shared::<f32, 256>();
    let lane = thread_idx().x;
    let block = block_idx().x;
    let input = input.as_slice();
    let i = thread_index().get();
    let value = if i < input.len() { input[i] } else { 0.0 };
    // SAFETY: lanes index the block's 256-element tile, and each block writes its own output.
    unsafe {
        shared.store_unchecked(lane, value);
        threadgroup::barrier();
        let mut stride = 128;
        while stride > 0 {
            if lane < stride {
                let sum = shared.load_unchecked(lane) + shared.load_unchecked(lane + stride);
                shared.store_unchecked(lane, sum);
            }
            threadgroup::barrier();
            stride >>= 1;
        }
        if lane == 0 && (block as usize) < output.len() {
            output.store_unchecked(block, shared.load_unchecked(0));
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use metal_oxide::{Device, LaunchConfig, Operation};
    let device = Device::system_default()?;
    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        let values = (0..n).map(|i| (i % 17) as f32 - 8.0).collect::<Vec<_>>();
        let input = device.buffer_from_slice(&values)?;
        let mut output = device.buffer_zeroed::<f32>(n.div_ceil(256) as usize)?;
        let config = LaunchConfig::<256>::for_elements(n)?;
        // SAFETY: nothing else writes the input while the kernel runs.
        unsafe { reduce(&device, config, &input, &mut output) }.wait()?;
        let expected = values
            .chunks(256)
            .map(|v| v.iter().sum::<f32>())
            .collect::<Vec<_>>();
        assert_eq!(output.read().wait(), expected, "n={n}");
        println!("reduction n={n}: verified on {}", device.name());
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("reduction runs on macOS with Apple Silicon");
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a Metal device"]
fn reduction_matches_cpu() {
    main().unwrap();
}
