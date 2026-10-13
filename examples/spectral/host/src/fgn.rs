use std::{
    future::{Future, IntoFuture},
    pin::Pin,
    sync::Arc,
    task::{Poll, Wake, Waker},
    time::Instant,
};

use metal_oxide::{
    Argument, Batch, Buffer, Device, Dim3, DynamicLaunchConfig, Launch, Module, Operation,
    Pipeline, sequence,
};
use serde::Serialize;

use crate::{Result, cpu, kernels::Readout};

#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub n: u32,
    pub size: u32,
    pub rows: u32,
    pub total: u32,
    pub parity: u32,
    pub base_cell: u64,
    pub seed: u32,
    pub scale: f32,
    pub tile: u32,
    pub stages: u32,
}

impl Plan {
    pub fn new(
        size: u32,
        rows: u32,
        first_row: u64,
        hurst: f64,
        seed: u32,
        max_threads: usize,
    ) -> Result<Self> {
        if rows > i32::MAX as u32 || !(0.0..=1.0).contains(&hurst) {
            return Err("invalid spectral shape or Hurst parameter".into());
        }
        let n = size
            .checked_next_power_of_two()
            .and_then(|n| n.checked_mul(2))
            .ok_or("transform extent overflow")?;
        let parity = (first_row & 1) as u32;
        let transforms = if rows == 0 {
            0
        } else {
            (rows + parity).div_ceil(2)
        };
        let total = transforms
            .checked_mul(n)
            .ok_or("transform count exceeds u32")?;
        rows.checked_mul(size).ok_or("output count exceeds u32")?;
        let log_n = n.trailing_zeros();
        let mut tile_log = log_n.min(11);
        while tile_log > 1 && (1_usize << (tile_log - 1)) > max_threads {
            tile_log -= 1;
        }
        if max_threads == 0 {
            return Err("pipeline supports no threads".into());
        }
        let mut stages = tile_log.min(log_n - 1);
        if (log_n - stages) % 2 == 1 && stages >= 1 {
            stages -= 1;
        }
        Ok(Self {
            n,
            size,
            rows,
            total,
            parity,
            base_cell: first_row.wrapping_sub(parity as u64).wrapping_mul(n as u64),
            seed,
            scale: (size.max(1) as f32).powf(-(hurst as f32)),
            tile: 1 << tile_log,
            stages,
        })
    }

    pub fn single_tail(self) -> bool {
        self.n.trailing_zeros() - self.stages == 1
    }
}

pub struct Buffers {
    pub real: Buffer<f32>,
    pub imag: Buffer<f32>,
    pub roots: Buffer<f32>,
    pub reverse: Buffer<u32>,
    pub output: Buffer<f32>,
    pub setup_ns: u64,
    pub allocation_ns: u64,
}

impl Buffers {
    pub fn new(device: &Device, plan: Plan, hurst: f64) -> Result<Self> {
        let start = Instant::now();
        let roots = cpu::eigenvalues(plan.n as usize, hurst);
        let reverse = cpu::reverse(plan.n as usize);
        let setup_ns = elapsed(start);
        let start = Instant::now();
        Ok(Self {
            real: device.buffer_zeroed(plan.total as usize)?,
            imag: device.buffer_zeroed(plan.total as usize)?,
            roots: device.buffer_from_slice(&roots)?,
            reverse: device.buffer_from_slice(&reverse)?,
            output: device.buffer_zeroed((plan.rows * plan.size) as usize)?,
            setup_ns,
            allocation_ns: elapsed(start),
        })
    }
}

#[derive(Default, Serialize)]
pub struct Timing {
    pub encode_ns: u64,
    pub submit_ns: u64,
    pub wait_ns: u64,
    pub gpu_ns: Option<u64>,
    pub readback_ns: u64,
    pub total_ns: u64,
}

pub fn elapsed(start: Instant) -> u64 {
    start.elapsed().as_nanos().try_into().unwrap()
}

pub struct Context {
    tile: Pipeline,
    radix4: Pipeline,
    extract: Pipeline,
    single: Pipeline,
    pub load_ns: u64,
    pub pipeline_ns: u64,
}

impl Context {
    pub fn new(device: &Device, directory: &str) -> Result<Self> {
        let start = Instant::now();
        let module = Module::from_artifact(device, directory)?;
        let load_ns = elapsed(start);
        let start = Instant::now();
        Ok(Self {
            tile: Pipeline::new(device, &module, "gen_tile_fft")?,
            radix4: Pipeline::new(device, &module, "fft_radix4")?,
            extract: Pipeline::new(device, &module, "fft_radix4_extract")?,
            single: Pipeline::new(device, &module, "fft_butterfly_extract")?,
            load_ns,
            pipeline_ns: elapsed(start),
        })
    }

    pub fn max_threads(&self) -> usize {
        self.tile.max_threads_per_block()
    }

    pub fn run(&self, plan: Plan, buffers: &mut Buffers) -> Result<Timing> {
        let mut timing = Timing::default();
        let start = Instant::now();
        let operation = sequence(|batch| {
            let encode = Instant::now();
            if plan.rows != 0 {
                // SAFETY: the plan bounds every allocation and dispatch. Tiles divide each
                // complete transform; every lane initializes scratch and reaches all barriers.
                // Ordered passes own disjoint butterflies, and extraction guards output rows.
                unsafe { self.encode(batch, plan, buffers)? };
            }
            timing.encode_ns = elapsed(encode);
            Ok(())
        });
        let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
        let mut cx = std::task::Context::from_waker(&waker);
        let mut future = operation.timed().into_future();
        let mut poll = Pin::new(&mut future).poll(&mut cx);
        timing.submit_ns = elapsed(start);
        let waiting = Instant::now();
        let ((), gpu) = loop {
            match poll {
                Poll::Ready(result) => break result?,
                Poll::Pending => {
                    std::thread::park();
                    poll = Pin::new(&mut future).poll(&mut cx);
                }
            }
        };
        timing.gpu_ns = gpu.map(|t| t.as_nanos().try_into().unwrap());
        timing.wait_ns = elapsed(waiting);
        let readback = Instant::now();
        std::hint::black_box(buffers.output.read().wait());
        timing.readback_ns = elapsed(readback);
        timing.total_ns = elapsed(start);
        Ok(timing)
    }

    /// # Safety
    ///
    /// The plan must describe the buffers; see [`Context::run`].
    unsafe fn encode(
        &self,
        batch: &mut Batch,
        plan: Plan,
        buffers: &mut Buffers,
    ) -> metal_oxide::Result<()> {
        let tile =
            DynamicLaunchConfig::new(Dim3::x(plan.total / plan.tile), Dim3::x(plan.tile / 2));
        // SAFETY: forwarded from the caller.
        unsafe {
            batch.push(Launch::new(
                Ok(&self.tile),
                tile,
                (
                    &mut buffers.real,
                    &mut buffers.imag,
                    &buffers.roots,
                    &buffers.reverse,
                    plan,
                ),
                |(real, imag, roots, reverse, plan)| {
                    Ok([
                        Argument::write(real),
                        Argument::write(imag),
                        Argument::read(roots),
                        Argument::read(reverse),
                        Argument::value(plan.n)?,
                        Argument::value(plan.seed)?,
                        Argument::value(plan.base_cell)?,
                        Argument::value(plan.stages)?,
                        Argument::threadgroup::<f32>(plan.tile as usize)?,
                        Argument::threadgroup::<f32>(plan.tile as usize)?,
                    ])
                },
            ))?;
            for stage in (plan.stages..plan.n.trailing_zeros().saturating_sub(2)).step_by(2) {
                batch.push(Launch::new(
                    Ok(&self.radix4),
                    DynamicLaunchConfig::for_elements(plan.total / 4, 256)?,
                    (&mut buffers.real, &mut buffers.imag, plan, stage),
                    |(real, imag, plan, stage)| {
                        Ok([
                            Argument::read_write(real),
                            Argument::read_write(imag),
                            Argument::value(plan.n)?,
                            Argument::value(*stage)?,
                            Argument::value(plan.total / 4)?,
                        ])
                    },
                ))?;
            }
            let (pipeline, count) = if plan.single_tail() {
                (&self.single, plan.total / 2)
            } else {
                (&self.extract, plan.total / 4)
            };
            batch.push(Launch::new(
                Ok(pipeline),
                DynamicLaunchConfig::for_elements(count, 256)?,
                (
                    &buffers.real,
                    &buffers.imag,
                    &mut buffers.output,
                    plan,
                    count,
                ),
                |(real, imag, output, plan, count)| {
                    Ok([
                        Argument::read(real),
                        Argument::read(imag),
                        Argument::value(plan.n)?,
                        Argument::write(output),
                        Argument::value(Readout {
                            size: plan.size,
                            scale: plan.scale,
                            rows: plan.rows,
                            parity: plan.parity,
                        })?,
                        Argument::value(*count)?,
                    ])
                },
            ))?;
        }
        Ok(())
    }
}

struct Unpark(std::thread::Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
