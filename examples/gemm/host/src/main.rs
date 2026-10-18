#[metal_oxide::kernels(path = "../kernels")]
mod kernels {}

/// (m, n, k) from one element to an 8x8 grid of whole 64x64 tiles, ragged in every dimension.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const SHAPES: [(u32, u32, u32); 7] = [
    (1, 1, 1),
    (8, 8, 8),
    (63, 65, 17),
    (64, 64, 64),
    (128, 96, 48),
    (257, 129, 95),
    (512, 512, 512),
];

/// The initial value of C and of 8 elements past it; no result of these inputs equals it.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const SENTINEL: f32 = -8192.0;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = metal_oxide::Device::system_default()?;
    verify(&device)?;
    Ok(())
}

/// Checks both kernels bitwise against the CPU on every shape; false reports a device without
/// SIMD-group matrices, which rejects loading the kernels.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn verify(device: &metal_oxide::Device) -> metal_oxide::Result<bool> {
    use metal_oxide::{Dim3, Error, F16, LaunchConfig, Operation};
    match kernels::module(device) {
        Ok(_) => {}
        Err(Error::UnsupportedDevice(reason)) if reason.contains("simdgroup_matrix") => {
            println!("gemm: {} is unsupported: {reason}", device.name());
            return Ok(false);
        }
        Err(error) => return Err(error),
    }
    let half = |values: &[f32]| values.iter().map(|&x| F16::from_f32(x)).collect::<Vec<_>>();
    for (m, n, k) in SHAPES {
        let (a, b) = inputs(m, n, k);
        let expected = reference(&a, &b, (m, n, k));
        let config = LaunchConfig::<128>::new(Dim3::xy(n.div_ceil(64), m.div_ceil(64)));
        let output = vec![SENTINEL; expected.len() + 8];
        let (input_a, input_b) = (device.buffer_from_slice(&a)?, device.buffer_from_slice(&b)?);
        let mut c = device.buffer_from_slice(&output)?;
        // SAFETY: distinct buffers cover m*k, k*n and m*n elements and the grid is the kernel's.
        unsafe { kernels::gemm_f32(device, config, &input_a, &input_b, &mut c, m, n, k) }.wait()?;
        check("gemm_f32", c.read().wait(), &expected, (m, n, k));
        let (input_a, input_b) = (
            device.buffer_from_slice(&half(&a))?,
            device.buffer_from_slice(&half(&b))?,
        );
        let mut c = device.buffer_from_slice(&output)?;
        // SAFETY: distinct buffers cover m*k, k*n and m*n elements and the grid is the kernel's.
        unsafe { kernels::gemm_f16(device, config, &input_a, &input_b, &mut c, m, n, k) }.wait()?;
        check("gemm_f16", c.read().wait(), &expected, (m, n, k));
        println!(
            "gemm {m}x{k} x {k}x{n}: f32 and F16 inputs verified on {}",
            device.name()
        );
    }
    Ok(true)
}

/// Multiples of 1/8 and 1/4, exact in F16 and in f32 sums of any order, then NaN as far as a K
/// step can read past k: 16 elements after A and 16 rows after B.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn inputs(m: u32, n: u32, k: u32) -> (Vec<f32>, Vec<f32>) {
    let a = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.125);
    let b = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.25);
    let nan = |count| std::iter::repeat_n(f32::NAN, count);
    let (a, b) = (a.chain(nan(16)), b.chain(nan(16 * n as usize)));
    (a.collect(), b.collect())
}

/// C = A * B accumulated in f64, which is exact for these inputs.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn reference(a: &[f32], b: &[f32], (m, n, k): (u32, u32, u32)) -> Vec<f32> {
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let mut c = vec![0.0_f64; m * n];
    for row in 0..m {
        for inner in 0..k {
            let x = f64::from(a[row * k + inner]);
            for column in 0..n {
                c[row * n + column] += x * f64::from(b[inner * n + column]);
            }
        }
    }
    c.into_iter().map(|value| value as f32).collect()
}

/// Compares C bitwise with the expected results and checks that the kernel wrote nothing past C.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn check(kernel: &str, actual: &[f32], expected: &[f32], (m, n, k): (u32, u32, u32)) {
    let (computed, past) = actual.split_at(expected.len());
    assert_eq!(past, [SENTINEL; 8], "{kernel} {m}x{n}x{k} wrote past C");
    let mut differs = computed.iter().zip(expected);
    if let Some(index) = differs.position(|(x, y)| x.to_bits() != y.to_bits()) {
        let (row, column) = (index as u32 / n, index as u32 % n);
        panic!(
            "{kernel} {m}x{n}x{k}: C[{row}][{column}] is {} instead of {}",
            computed[index], expected[index]
        );
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("gemm runs on macOS with Apple Silicon");
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a Metal device"]
fn generated_gemm_matches_cpu() {
    use objc2_metal::{MTLDevice, MTLGPUFamily};
    let device = metal_oxide::Device::system_default().unwrap();
    // Apple7 and later GPUs must run the kernels; only other devices may reject them.
    let apple7 = device.as_native().supportsFamily(MTLGPUFamily::Apple7);
    assert_eq!(verify(&device).unwrap(), apple7);
}
