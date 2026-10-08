use crate::kernels::Kernels;
use metal_oxide::{Batch, Buffer, Device, LaunchConfig, Result};

struct Level {
    local: Buffer<u32>,
    output: Buffer<u32>,
    totals: Buffer<u32>,
    n: u32,
}

pub(super) struct Scan {
    levels: Vec<Level>,
    zero: Buffer<u32>,
}

impl Scan {
    pub(super) fn new(device: &Device, mut n: u32) -> Result<Self> {
        let mut levels = Vec::new();
        loop {
            let blocks = n.div_ceil(64);
            levels.push(Level {
                local: device.buffer_zeroed(n as usize)?,
                output: device.buffer_zeroed(n as usize)?,
                totals: device.buffer_zeroed(blocks as usize)?,
                n,
            });
            if blocks <= 1 {
                break;
            }
            n = blocks;
        }
        Ok(Self {
            levels,
            zero: device.buffer_zeroed(1)?,
        })
    }

    pub(super) fn output(&self) -> &Buffer<u32> {
        &self.levels[0].output
    }

    pub(super) unsafe fn enqueue(
        &mut self,
        batch: &mut Batch<'_>,
        kernels: &Kernels<'_>,
        input: &Buffer<u32>,
    ) -> Result<()> {
        // SAFETY: each level owns disjoint buffers sized for its input and block count.
        unsafe {
            let first = &mut self.levels[0];
            kernels.enqueue_block_scan(
                batch,
                LaunchConfig::<64>::for_elements(first.n)?,
                input,
                &mut first.local,
                &mut first.totals,
                first.n,
            )?;
            for index in 1..self.levels.len() {
                let (previous, current) = self.levels.split_at_mut(index);
                let previous = &previous[index - 1];
                let current = &mut current[0];
                kernels.enqueue_block_scan(
                    batch,
                    LaunchConfig::<64>::for_elements(current.n)?,
                    &previous.totals,
                    &mut current.local,
                    &mut current.totals,
                    current.n,
                )?;
            }
            let root = self.levels.last_mut().unwrap();
            kernels.enqueue_add_offsets(
                batch,
                LaunchConfig::<64>::for_elements(root.n)?,
                &root.local,
                &self.zero,
                &mut root.output,
                root.n,
            )?;
            for index in (0..self.levels.len() - 1).rev() {
                let (child, parent) = self.levels.split_at_mut(index + 1);
                let child = &mut child[index];
                kernels.enqueue_add_offsets(
                    batch,
                    LaunchConfig::<64>::for_elements(child.n)?,
                    &child.local,
                    &parent[0].output,
                    &mut child.output,
                    child.n,
                )?;
            }
        }
        Ok(())
    }
}
