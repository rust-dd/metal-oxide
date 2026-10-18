use super::*;
use metal_oxide::{Argument, Dim3, F16, GpuValue, LaunchConfig};

mod mps;
#[cfg(test)]
mod tests;

/// The inputs of one m x n x k shape and their exact product, which all cases share.
pub(super) struct Problem {
    shape: [u32; 3],
    a: Vec<f32>,
    b: Vec<f32>,
    expected: Vec<f32>,
}

impl Problem {
    /// Multiples of 1/8 and 1/4 make every product and partial sum exact in f32, in any order.
    pub(super) fn new(shape: [u32; 3]) -> Self {
        let [m, n, k] = shape.map(|dimension| dimension as usize);
        let a = (0..m * k)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
            .collect::<Vec<_>>();
        let b = (0..k * n)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.25)
            .collect::<Vec<_>>();
        let expected = product(&a, &b, [m, n, k]);
        Self {
            shape,
            a,
            b,
            expected,
        }
    }

    /// A and B as F16, which holds every input exactly.
    fn half(&self) -> [Vec<F16>; 2] {
        [&self.a, &self.b].map(|values| values.iter().map(|&x| F16::from_f32(x)).collect())
    }
}

/// Row-major A * B accumulated in f64, which is exact for the inputs of [`Problem`].
fn product(a: &[f32], b: &[f32], [m, n, k]: [usize; 3]) -> Vec<f32> {
    assert_eq!((a.len(), b.len()), (m * k, k * n));
    let mut c = vec![0.0_f64; m * n];
    for (a_row, c_row) in a.chunks_exact(k).zip(c.chunks_exact_mut(n)) {
        for (&x, b_row) in a_row.iter().zip(b.chunks_exact(n)) {
            for (sum, &y) in c_row.iter_mut().zip(b_row) {
                *sum += f64::from(x) * f64::from(y);
            }
        }
    }
    c.into_iter().map(|sum| sum as f32).collect()
}

/// Compares C bitwise with the expected result and names the first element that differs.
fn check(case: &str, actual: &[f32], expected: &[f32], columns: u32) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{case}: C has {} elements, not {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    match (actual.iter().zip(expected)).position(|(x, y)| x.to_bits() != y.to_bits()) {
        None => Ok(()),
        Some(index) => {
            let (row, column) = (index / columns as usize, index % columns as usize);
            Err(format!(
                "{case}: C[{row}][{column}] is {} instead of {}",
                actual[index], expected[index]
            )
            .into())
        }
    }
}

/// Finishes `case` and adds the GPU and host times of one GEMM: a sample's GPU time and its
/// submit plus wait time, each over its `repeat`.
fn finish(case: Case) -> Result<Case> {
    let mut case = case.finish()?;
    let repeat = u64::from(case.repeat);
    let per_gemm = |ns: u64| (ns + repeat / 2) / repeat;
    let gpu = case.samples.iter().filter_map(|sample| sample.gpu_ns);
    let gpu = gpu.map(per_gemm).collect::<Vec<_>>();
    if !gpu.is_empty() {
        case.summary_ns
            .insert("gpu_per_gemm_ns".into(), Summary::new(&gpu)?);
    }
    let host = case.samples.iter();
    let host = host
        .map(|sample| per_gemm(sample.submit_ns + sample.wait_ns))
        .collect::<Vec<_>>();
    case.summary_ns
        .insert("host_per_gemm_ns".into(), Summary::new(&host)?);
    Ok(case)
}

pub(super) fn run(
    device: &Device,
    config: &Config,
    problem: &Problem,
    variant: &str,
) -> Result<Vec<Case>> {
    if variant == "mps" {
        return mps::run(device, config, problem);
    }
    let [a, b] = problem.half();
    Ok(vec![
        measure(
            device,
            config,
            problem,
            variant,
            "f32",
            (&problem.a, &problem.b),
        )?,
        measure(device, config, problem, variant, "f16", (&a, &b))?,
    ])
}

fn measure<T: GpuValue>(
    device: &Device,
    config: &Config,
    problem: &Problem,
    variant: &str,
    input: &str,
    (a_values, b_values): (&[T], &[T]),
) -> Result<Case> {
    let [m, n, k] = problem.shape;
    let name = format!("gemm-{input}");
    let mut case = Case::new(&name, variant, problem.shape.to_vec());
    case.repeat = config.gemm_repeat;
    let entry = if variant == "rust" {
        format!("gemm_{input}")
    } else {
        format!("gemm_{input}_reference")
    };
    let pipelines = case.pipelines(device, config, "gemm", &[&entry])?;
    let start = Instant::now();
    let mut a = device.buffer_zeroed::<T>(a_values.len())?;
    let mut b = device.buffer_zeroed::<T>(b_values.len())?;
    let mut c = device.buffer_zeroed::<f32>(problem.expected.len())?;
    case.allocation_ns = elapsed(start);
    let launch = LaunchConfig::<128>::new(Dim3::xy(n.div_ceil(64), m.div_ceil(64)));
    for iteration in 0..config.warmup + config.samples {
        let start = Instant::now();
        a.write().wait().copy_from_slice(a_values);
        b.write().wait().copy_from_slice(b_values);
        let copy = elapsed(start);
        c.write().wait().fill(f32::NAN);
        let start = Instant::now();
        a.upload();
        b.upload();
        c.upload();
        let upload = elapsed(start);
        let mut sample = dispatch(|batch| {
            let (mut left, mut right, mut out) = (&a, &b, &mut c);
            // SAFETY: distinct buffers cover m*k, k*n and m*n elements, the grid is the kernel's,
            // and dimensions of at most 8192 keep every index in u32.
            unsafe {
                for _ in 0..config.gemm_repeat {
                    (left, right, out, ..) = batch.push(Launch::new(
                        Ok(&pipelines[0]),
                        launch,
                        (left, right, out, m, n, k),
                        |(a, b, c, m, n, k)| {
                            Ok([
                                Argument::read(a),
                                Argument::read(b),
                                Argument::write(c),
                                Argument::value(*m)?,
                                Argument::value(*n)?,
                                Argument::value(*k)?,
                            ])
                        },
                    ))?;
                }
            }
            Ok(())
        })?;
        let start = Instant::now();
        let result = c.read().wait();
        sample.readback_ns = elapsed(start);
        check(&format!("{name} {variant}"), result, &problem.expected, n)?;
        sample.input_copy_ns = copy;
        sample.upload_ns = upload;
        if iteration >= config.warmup {
            case.samples.push(sample);
        }
    }
    finish(case)
}
