use metal_oxide::{Argument, Batch, Buffer, Device, LaunchConfig, Module, Pipeline, Result};

pub(super) enum Program<'d> {
    Generated(super::kernels::Kernels<'d>),
    Reference(Reference),
}
pub(super) struct Reference {
    histogram: Pipeline,
    block_scan: Pipeline,
    add_offsets: Pipeline,
    mark: Pipeline,
    scatter: Pipeline,
}
impl<'d> Program<'d> {
    pub(super) fn generated(device: &'d Device) -> Result<Self> {
        Ok(Self::Generated(super::kernels::load(
            device,
            env!("METAL_OXIDE_ARTIFACT_DIR"),
        )?))
    }
    pub(super) fn reference(device: &'d Device) -> Result<Self> {
        let module = Module::from_source(
            device,
            include_str!("../../../../benchmarks/reference-msl/cooperation.metal"),
        )?;
        Ok(Self::Reference(Reference {
            histogram: Pipeline::new(device, &module, "histogram")?,
            block_scan: Pipeline::new(device, &module, "block_scan")?,
            add_offsets: Pipeline::new(device, &module, "add_offsets")?,
            mark: Pipeline::new(device, &module, "mark")?,
            scatter: Pipeline::new(device, &module, "scatter")?,
        }))
    }
    pub(super) unsafe fn histogram(
        &self,
        batch: &mut Batch<'_>,
        input: &Buffer<u32>,
        bins: &mut Buffer<u32>,
        n: u32,
    ) -> Result<()> {
        let config = LaunchConfig::<256>::for_elements(n)?;
        // SAFETY: the workload supplies matching bounds, initialized resources, and ordered passes.
        unsafe {
            match self {
                Self::Generated(kernels) => {
                    kernels.enqueue_histogram(batch, config, input, bins, n)
                }
                Self::Reference(kernels) => batch.launch(
                    &kernels.histogram,
                    config,
                    &[
                        Argument::read(input),
                        Argument::atomic(bins),
                        Argument::value(n)?,
                    ],
                ),
            }
        }
    }
    pub(super) unsafe fn block_scan(
        &self,
        batch: &mut Batch<'_>,
        input: &Buffer<u32>,
        prefix: &mut Buffer<u32>,
        totals: &mut Buffer<u32>,
        n: u32,
    ) -> Result<()> {
        let config = LaunchConfig::<256>::for_elements(n)?;
        // SAFETY: the workload supplies matching bounds, initialized resources, and ordered passes.
        unsafe {
            match self {
                Self::Generated(kernels) => {
                    kernels.enqueue_block_scan(batch, config, input, prefix, totals, n)
                }
                Self::Reference(kernels) => batch.launch(
                    &kernels.block_scan,
                    config,
                    &[
                        Argument::read(input),
                        Argument::write(prefix),
                        Argument::write(totals),
                        Argument::value(n)?,
                    ],
                ),
            }
        }
    }
    pub(super) unsafe fn add_offsets(
        &self,
        batch: &mut Batch<'_>,
        local: &Buffer<u32>,
        offsets: &Buffer<u32>,
        output: &mut Buffer<u32>,
        n: u32,
    ) -> Result<()> {
        let config = LaunchConfig::<256>::for_elements(n)?;
        // SAFETY: the workload supplies matching bounds, initialized resources, and ordered passes.
        unsafe {
            match self {
                Self::Generated(kernels) => {
                    kernels.enqueue_add_offsets(batch, config, local, offsets, output, n)
                }
                Self::Reference(kernels) => batch.launch(
                    &kernels.add_offsets,
                    config,
                    &[
                        Argument::read(local),
                        Argument::read(offsets),
                        Argument::write(output),
                        Argument::value(n)?,
                    ],
                ),
            }
        }
    }
    pub(super) unsafe fn mark(
        &self,
        batch: &mut Batch<'_>,
        input: &Buffer<u32>,
        flags: &mut Buffer<u32>,
        n: u32,
    ) -> Result<()> {
        let config = LaunchConfig::<256>::for_elements(n)?;
        // SAFETY: the workload supplies matching bounds, initialized resources, and ordered passes.
        unsafe {
            match self {
                Self::Generated(kernels) => kernels.enqueue_mark(batch, config, input, flags, n),
                Self::Reference(kernels) => batch.launch(
                    &kernels.mark,
                    config,
                    &[
                        Argument::read(input),
                        Argument::write(flags),
                        Argument::value(n)?,
                    ],
                ),
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn scatter(
        &self,
        batch: &mut Batch<'_>,
        input: &Buffer<u32>,
        flags: &Buffer<u32>,
        prefix: &Buffer<u32>,
        output: &mut Buffer<u32>,
        count: &mut Buffer<u32>,
        n: u32,
    ) -> Result<()> {
        let config = LaunchConfig::<256>::for_elements(n)?;
        // SAFETY: the workload supplies matching bounds, initialized resources, and ordered passes.
        unsafe {
            match self {
                Self::Generated(kernels) => {
                    kernels.enqueue_scatter(batch, config, input, flags, prefix, output, count, n)
                }
                Self::Reference(kernels) => batch.launch(
                    &kernels.scatter,
                    config,
                    &[
                        Argument::read(input),
                        Argument::read(flags),
                        Argument::read(prefix),
                        Argument::write(output),
                        Argument::write(count),
                        Argument::value(n)?,
                    ],
                ),
            }
        }
    }
}
