use std::time::Instant;

use metal_oxide::{Argument, Buffer, Device, Dim3, DynamicLaunchConfig, Module, Pipeline};
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

    pub fn run(&self, device: &Device, plan: Plan, buffers: &mut Buffers) -> Result<Timing> {
        let mut timing = Timing::default();
        let start = Instant::now();
        // SAFETY: the plan bounds every allocation and dispatch. Tiles divide each
        // complete transform; every lane initializes scratch and reaches all barriers.
        // Ordered passes own disjoint butterflies, and extraction guards output rows.
        let submission = unsafe {
            device.submit(|batch| {
                let encode = Instant::now();
                if plan.rows != 0 {
                    batch.launch(
                        &self.tile,
                        DynamicLaunchConfig::new(
                            Dim3::x(plan.total / plan.tile),
                            Dim3::x(plan.tile / 2),
                        ),
                        &[
                            Argument::write(&mut buffers.real),
                            Argument::write(&mut buffers.imag),
                            Argument::read(&buffers.roots),
                            Argument::read(&buffers.reverse),
                            Argument::value(plan.n)?,
                            Argument::value(plan.seed)?,
                            Argument::value(plan.base_cell)?,
                            Argument::value(plan.stages)?,
                            Argument::threadgroup::<f32>(plan.tile as usize)?,
                            Argument::threadgroup::<f32>(plan.tile as usize)?,
                        ],
                    )?;
                    for stage in (plan.stages..plan.n.trailing_zeros().saturating_sub(2)).step_by(2)
                    {
                        batch.launch(
                            &self.radix4,
                            DynamicLaunchConfig::for_elements(plan.total / 4, 256)?,
                            &[
                                Argument::read_write(&mut buffers.real),
                                Argument::read_write(&mut buffers.imag),
                                Argument::value(plan.n)?,
                                Argument::value(stage)?,
                                Argument::value(plan.total / 4)?,
                            ],
                        )?;
                    }
                    let (pipeline, count) = if plan.single_tail() {
                        (&self.single, plan.total / 2)
                    } else {
                        (&self.extract, plan.total / 4)
                    };
                    batch.launch(
                        pipeline,
                        DynamicLaunchConfig::for_elements(count, 256)?,
                        &[
                            Argument::read(&buffers.real),
                            Argument::read(&buffers.imag),
                            Argument::value(plan.n)?,
                            Argument::write(&mut buffers.output),
                            Argument::value(Readout {
                                size: plan.size,
                                scale: plan.scale,
                                rows: plan.rows,
                                parity: plan.parity,
                            })?,
                            Argument::value(count)?,
                        ],
                    )?;
                }
                timing.encode_ns = elapsed(encode);
                Ok(())
            })?
        };
        timing.submit_ns = elapsed(start);
        let waiting = Instant::now();
        timing.gpu_ns = submission
            .wait()?
            .gpu_duration
            .map(|t| t.as_nanos().try_into().unwrap());
        timing.wait_ns = elapsed(waiting);
        let readback = Instant::now();
        std::hint::black_box(buffers.output.as_slice());
        timing.readback_ns = elapsed(readback);
        timing.total_ns = elapsed(start);
        Ok(timing)
    }
}
