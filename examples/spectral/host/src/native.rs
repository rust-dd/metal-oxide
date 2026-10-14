use std::{ptr::NonNull, time::Instant};

use metal_oxide::Device;
use metal_oxide_artifact::MathMode;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLMathFloatingPointFunctions, MTLMathMode, MTLSize,
};

use crate::{
    Result,
    fgn::{self, Plan, Timing, elapsed},
};

type MetalBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type MetalPipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub struct Buffers {
    real: MetalBuffer,
    imag: MetalBuffer,
    roots: MetalBuffer,
    reverse: MetalBuffer,
    output: MetalBuffer,
    len: usize,
    pub setup_ns: u64,
    pub allocation_ns: u64,
}

impl From<fgn::Buffers> for Buffers {
    fn from(buffers: fgn::Buffers) -> Self {
        Self {
            len: buffers.output.len(),
            real: buffers.real.into_native(),
            imag: buffers.imag.into_native(),
            roots: buffers.roots.into_native(),
            reverse: buffers.reverse.into_native(),
            output: buffers.output.into_native(),
            setup_ns: buffers.setup_ns,
            allocation_ns: buffers.allocation_ns,
        }
    }
}

impl Buffers {
    pub fn output(&self) -> &[f32] {
        // SAFETY: owning exports initialize these f32 values, run waits before returning,
        // and its exclusive borrow prevents views from crossing a native GPU write.
        unsafe {
            std::slice::from_raw_parts(self.output.contents().cast::<f32>().as_ptr(), self.len)
        }
    }

    pub fn into_output(self) -> MetalBuffer {
        self.output
    }
}

pub struct Context {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    tile: MetalPipeline,
    radix4: MetalPipeline,
    extract: MetalPipeline,
    single: MetalPipeline,
    pub load_ns: u64,
    pub pipeline_ns: u64,
}

impl Context {
    pub fn new(device: &Device, mode: MathMode) -> Result<Self> {
        let start = Instant::now();
        let options = MTLCompileOptions::new();
        options.setMathMode(if mode == MathMode::Precise {
            MTLMathMode::Safe
        } else {
            MTLMathMode::Relaxed
        });
        options.setMathFloatingPointFunctions(if mode == MathMode::Precise {
            MTLMathFloatingPointFunctions::Precise
        } else {
            MTLMathFloatingPointFunctions::Fast
        });
        let pragma = if mode == MathMode::Precise {
            "#pragma STDC FP_CONTRACT OFF\n"
        } else {
            "#pragma METAL fp contract(fast)\n"
        };
        let source = format!(
            "{pragma}{}\n{}",
            include_str!("../common.metal"),
            include_str!("../fgn.metal")
        );
        let native = device.as_native();
        let library = native
            .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
            .map_err(|e| e.localizedDescription().to_string())?;
        let load_ns = elapsed(start);
        let start = Instant::now();
        let pipeline = |name| -> Result<MetalPipeline> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or("missing native kernel")?;
            Ok(native
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| e.localizedDescription().to_string())?)
        };
        Ok(Self {
            queue: native
                .newCommandQueue()
                .ok_or("native queue allocation failed")?,
            tile: pipeline("gen_tile_fft")?,
            radix4: pipeline("fft_radix4")?,
            extract: pipeline("fft_radix4_extract")?,
            single: pipeline("fft_butterfly_extract")?,
            load_ns,
            pipeline_ns: elapsed(start),
        })
    }

    pub fn max_threads(&self) -> usize {
        self.tile.maxTotalThreadsPerThreadgroup()
    }

    pub fn run(&self, plan: Plan, buffers: &mut Buffers) -> Result<Timing> {
        if plan.rows == 0 {
            return Ok(Timing::default());
        }
        let start = Instant::now();
        let command = self
            .queue
            .commandBuffer()
            .ok_or("native command allocation failed")?;
        let encode = Instant::now();
        // SAFETY: all native allocations came from initialized canonical runtime buffers.
        // The same plan sizes their disjoint accesses and exact native dispatch counts.
        unsafe {
            let encoder = command
                .computeCommandEncoder()
                .ok_or("native encoder allocation failed")?;
            encoder.setComputePipelineState(&self.tile);
            bind(
                &encoder,
                &[
                    &buffers.real,
                    &buffers.imag,
                    &buffers.roots,
                    &buffers.reverse,
                ],
            );
            value(&encoder, 4, &plan.n);
            value(&encoder, 5, &plan.seed);
            value(&encoder, 6, &plan.base_cell);
            value(&encoder, 7, &plan.stages);
            // Metal requires threadgroup lengths in multiples of 16 bytes.
            let shared = (plan.tile as usize * 4).next_multiple_of(16);
            encoder.setThreadgroupMemoryLength_atIndex(shared, 0);
            encoder.setThreadgroupMemoryLength_atIndex(shared, 1);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                size(plan.total / plan.tile),
                size(plan.tile / 2),
            );
            encoder.endEncoding();
            for stage in (plan.stages..plan.n.trailing_zeros().saturating_sub(2)).step_by(2) {
                let encoder = command
                    .computeCommandEncoder()
                    .ok_or("native encoder allocation failed")?;
                encoder.setComputePipelineState(&self.radix4);
                bind(&encoder, &[&buffers.real, &buffers.imag]);
                value(&encoder, 2, &plan.n);
                value(&encoder, 3, &stage);
                encoder.dispatchThreads_threadsPerThreadgroup(size(plan.total / 4), size(256));
                encoder.endEncoding();
            }
            let encoder = command
                .computeCommandEncoder()
                .ok_or("native encoder allocation failed")?;
            encoder.setComputePipelineState(if plan.single_tail() {
                &self.single
            } else {
                &self.extract
            });
            bind(&encoder, &[&buffers.real, &buffers.imag]);
            value(&encoder, 2, &plan.n);
            encoder.setBuffer_offset_atIndex(Some(&buffers.output), 0, 3);
            value(&encoder, 4, &plan.size);
            value(&encoder, 5, &plan.scale);
            value(&encoder, 6, &plan.rows);
            value(&encoder, 7, &plan.parity);
            encoder.dispatchThreads_threadsPerThreadgroup(
                size(plan.total / if plan.single_tail() { 2 } else { 4 }),
                size(256),
            );
            encoder.endEncoding();
        }
        let encode_ns = elapsed(encode);
        command.commit();
        let submit_ns = elapsed(start);
        let waiting = Instant::now();
        command.waitUntilCompleted();
        let wait_ns = elapsed(waiting);
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!("native command failed: {:?}", command.error()).into());
        }
        let gpu_seconds = command.GPUEndTime() - command.GPUStartTime();
        let readback = Instant::now();
        std::hint::black_box(buffers.output());
        Ok(Timing {
            encode_ns,
            submit_ns,
            wait_ns,
            gpu_ns: (gpu_seconds > 0.0).then_some((gpu_seconds * 1e9) as u64),
            readback_ns: elapsed(readback),
            total_ns: elapsed(start),
        })
    }
}

fn size(width: u32) -> MTLSize {
    MTLSize {
        width: width as usize,
        height: 1,
        depth: 1,
    }
}

unsafe fn bind(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, buffers: &[&MetalBuffer]) {
    for (index, buffer) in buffers.iter().enumerate() {
        // SAFETY: the caller supplies the pipeline's contiguous buffer slots and live storage.
        unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
    }
}

unsafe fn value<T: Copy>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: usize,
    value: &T,
) {
    // SAFETY: only initialized primitive u32/u64/f32 values are passed; Metal copies
    // these bytes before returning, and the caller matches the native scalar binding.
    unsafe { encoder.setBytes_length_atIndex(NonNull::from(value).cast(), size_of::<T>(), index) };
}
