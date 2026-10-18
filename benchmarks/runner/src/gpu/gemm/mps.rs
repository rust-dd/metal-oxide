use std::time::Duration;

use objc2::{
    AnyThread,
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLDevice,
    MTLResourceOptions,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};

use super::*;

#[cfg(test)]
mod tests;

/// A matrix in shared storage whose rows have the stride MPS recommends for its type.
struct Matrix {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    matrix: Retained<MPSMatrix>,
    row_bytes: usize,
}

impl Matrix {
    fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        [rows, columns]: [usize; 2],
        data_type: MPSDataType,
    ) -> Result<Self> {
        // SAFETY: a class method without preconditions.
        let row_bytes =
            unsafe { MPSMatrixDescriptor::rowBytesForColumns_dataType(columns, data_type) };
        let buffer = device
            .newBufferWithLength_options(rows * row_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("MPS matrix allocation failed")?;
        // SAFETY: the descriptor states the layout of the `rows * row_bytes` bytes of the buffer.
        let matrix = unsafe {
            let descriptor =
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    rows, columns, row_bytes, data_type,
                );
            MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &buffer, &descriptor)
        };
        Ok(Self {
            buffer,
            matrix,
            row_bytes,
        })
    }

    /// The bytes of every row, padding included; the GPU must be idle while they are in use.
    fn bytes(&mut self) -> &mut [u8] {
        let (contents, length) = (self.buffer.contents(), self.buffer.length());
        // SAFETY: shared storage is CPU-visible, and each sample waits for its GPU work first.
        unsafe { std::slice::from_raw_parts_mut(contents.as_ptr().cast(), length) }
    }
}

/// An input element type and the MPS data type that names it.
trait Element: Copy {
    const DATA_TYPE: MPSDataType;

    fn bytes(self) -> impl IntoIterator<Item = u8>;
}

impl Element for f32 {
    const DATA_TYPE: MPSDataType = MPSDataType::Float32;

    fn bytes(self) -> impl IntoIterator<Item = u8> {
        self.to_ne_bytes()
    }
}

impl Element for F16 {
    const DATA_TYPE: MPSDataType = MPSDataType::Float16;

    fn bytes(self) -> impl IntoIterator<Item = u8> {
        self.to_bits().to_ne_bytes()
    }
}

/// `tight`, the rows of a matrix without padding, with each row zero-filled up to `row_bytes`.
fn pad_rows(tight: &[u8], rows: usize, row_bytes: usize) -> Vec<u8> {
    let row_len = tight.len() / rows;
    let mut padded = vec![0; rows * row_bytes];
    for (row, out) in tight
        .chunks_exact(row_len)
        .zip(padded.chunks_exact_mut(row_bytes))
    {
        out[..row_len].copy_from_slice(row);
    }
    padded
}

/// The f32 values of the first `columns` elements of every `row_bytes` row; padding is skipped.
fn unpad_rows(bytes: &[u8], columns: usize, row_bytes: usize, out: &mut Vec<f32>) {
    out.clear();
    for row in bytes.chunks_exact(row_bytes) {
        let (values, _) = row[..columns * 4].as_chunks::<4>();
        out.extend(values.iter().map(|&value| f32::from_ne_bytes(value)));
    }
}

/// The GPU time between two command buffer timestamps; absent when either is unavailable.
fn gpu_ns(start: f64, end: f64) -> Option<u64> {
    let valid = start.is_finite() && start > 0.0 && end.is_finite() && end >= start;
    let time = valid.then(|| Duration::try_from_secs_f64(end - start).ok());
    time.flatten()?.as_nanos().try_into().ok()
}

/// A case without a threadgroup size, since MPS picks its own kernels.
fn new_case(name: &str, shape: [u32; 3], repeat: u32) -> Case {
    let mut case = Case::new(name, "mps", shape.to_vec());
    case.repeat = repeat;
    case.block = None;
    case
}

/// Encodes `repeat` multiplications into one command buffer on `queue` and waits for it.
fn submit(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    kernel: &MPSMatrixMultiplication,
    [a, b, c]: [&MPSMatrix; 3],
    repeat: u32,
) -> Result<Sample> {
    autoreleasepool(|_| {
        let mut sample = Sample::default();
        let start = Instant::now();
        let command = queue
            .commandBuffer()
            .ok_or("MPS command buffer creation failed")?;
        let encoding = Instant::now();
        // Hazard tracking of the write to C serializes the repeats, as the batch orders ours.
        for _ in 0..repeat {
            // SAFETY: the matrices have the shapes the kernel was created for, and C is distinct.
            unsafe {
                kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(&command, a, b, c)
            };
        }
        sample.encode_ns = elapsed(encoding);
        command.commit();
        sample.submit_ns = elapsed(start);
        let start = Instant::now();
        command.waitUntilCompleted();
        sample.wait_ns = elapsed(start);
        if command.status() != MTLCommandBufferStatus::Completed {
            let reason = command
                .error()
                .map(|e| e.localizedDescription().to_string());
            return Err(format!(
                "MPS command buffer ended as {:?}: {}",
                command.status(),
                reason.unwrap_or_default()
            )
            .into());
        }
        sample.gpu_ns = gpu_ns(command.GPUStartTime(), command.GPUEndTime());
        Ok(sample)
    })
}

pub(super) fn run(device: &Device, config: &Config, problem: &Problem) -> Result<Vec<Case>> {
    let [a, b] = problem.half();
    Ok(vec![
        measure(device, config, problem, "f32", (&problem.a, &problem.b))?,
        measure(device, config, problem, "f16", (&a, &b))?,
    ])
}

fn measure<T: Element>(
    device: &Device,
    config: &Config,
    problem: &Problem,
    input: &str,
    (a_values, b_values): (&[T], &[T]),
) -> Result<Case> {
    let [m, n, k] = problem.shape.map(|dimension| dimension as usize);
    let name = format!("gemm-{input}");
    let mut case = new_case(&name, problem.shape, config.gemm_repeat);
    let native = device.as_native();
    let queue = native
        .newCommandQueue()
        .ok_or("MPS command queue creation failed")?;
    let start = Instant::now();
    // SAFETY: the kernel is created for the shapes of the matrices below.
    let kernel = unsafe {
        MPSMatrixMultiplication::initWithDevice_resultRows_resultColumns_interiorColumns(
            MPSMatrixMultiplication::alloc(),
            native,
            m,
            n,
            k,
        )
    };
    case.pipeline_ns = elapsed(start);
    let start = Instant::now();
    let mut a = Matrix::new(native, [m, k], T::DATA_TYPE)?;
    let mut b = Matrix::new(native, [k, n], T::DATA_TYPE)?;
    let mut c = Matrix::new(native, [m, n], MPSDataType::Float32)?;
    case.allocation_ns = elapsed(start);
    let tight = |values: &[T]| values.iter().flat_map(|&x| x.bytes()).collect::<Vec<_>>();
    let a_values = pad_rows(&tight(a_values), m, a.row_bytes);
    let b_values = pad_rows(&tight(b_values), k, b.row_bytes);
    let mut result = Vec::with_capacity(m * n);
    for iteration in 0..config.warmup + config.samples {
        let start = Instant::now();
        a.bytes().copy_from_slice(&a_values);
        b.bytes().copy_from_slice(&b_values);
        let copy = elapsed(start);
        // 0xFF bytes read as f32 NaN, so an element MPS leaves unwritten fails the check.
        c.bytes().fill(0xFF);
        let matrices = [&*a.matrix, &*b.matrix, &*c.matrix];
        let mut sample = submit(&queue, &kernel, matrices, config.gemm_repeat)?;
        let row_bytes = c.row_bytes;
        let start = Instant::now();
        let bytes = c.bytes();
        sample.readback_ns = elapsed(start);
        unpad_rows(bytes, n, row_bytes, &mut result);
        check(
            &format!("{name} mps"),
            &result,
            &problem.expected,
            problem.shape[1],
        )?;
        sample.input_copy_ns = copy;
        if iteration >= config.warmup {
            case.samples.push(sample);
        }
    }
    finish(case)
}
