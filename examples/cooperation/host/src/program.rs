use metal_oxide::{
    Argument, Batch, Buffer, Device, Launch, LaunchConfig, Module, Pipeline, Result,
};

pub(super) enum Program {
    Generated(Device),
    Reference(Reference),
}
pub(super) struct Reference {
    histogram: Pipeline,
    block_scan: Pipeline,
    add_offsets: Pipeline,
    mark: Pipeline,
    scatter: Pipeline,
}
impl Program {
    pub(super) fn generated(device: &Device) -> Self {
        Self::Generated(device.clone())
    }
    pub(super) fn reference(device: &Device) -> Result<Self> {
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
        batch: &mut Batch,
        input: &Buffer<u32>,
        bins: &mut Buffer<u32>,
        n: u32,
    ) -> Result<()> {
        let config = LaunchConfig::<256>::for_elements(n)?;
        // SAFETY: the workload supplies matching bounds, initialized resources, and ordered passes.
        unsafe {
            match self {
                Self::Generated(device) => batch
                    .push(super::kernels::histogram(device, config, input, bins, n))
                    .map(drop),
                Self::Reference(kernels) => batch
                    .push(Launch::new(
                        Ok(&kernels.histogram),
                        config,
                        (input, bins, n),
                        |(input, bins, n)| {
                            Ok([
                                Argument::read(input),
                                Argument::atomic(bins),
                                Argument::value(*n)?,
                            ])
                        },
                    ))
                    .map(drop),
            }
        }
    }
    pub(super) unsafe fn block_scan(
        &self,
        batch: &mut Batch,
        input: &Buffer<u32>,
        prefix: &mut Buffer<u32>,
        totals: &mut Buffer<u32>,
        n: u32,
    ) -> Result<()> {
        let config = LaunchConfig::<256>::for_elements(n)?;
        // SAFETY: the workload supplies matching bounds, initialized resources, and ordered passes.
        unsafe {
            match self {
                Self::Generated(device) => batch
                    .push(super::kernels::block_scan(
                        device, config, input, prefix, totals, n,
                    ))
                    .map(drop),
                Self::Reference(kernels) => batch
                    .push(Launch::new(
                        Ok(&kernels.block_scan),
                        config,
                        (input, prefix, totals, n),
                        |(input, prefix, totals, n)| {
                            Ok([
                                Argument::read(input),
                                Argument::write(prefix),
                                Argument::write(totals),
                                Argument::value(*n)?,
                            ])
                        },
                    ))
                    .map(drop),
            }
        }
    }
    pub(super) unsafe fn add_offsets(
        &self,
        batch: &mut Batch,
        local: &Buffer<u32>,
        offsets: &Buffer<u32>,
        output: &mut Buffer<u32>,
        n: u32,
    ) -> Result<()> {
        let config = LaunchConfig::<256>::for_elements(n)?;
        // SAFETY: the workload supplies matching bounds, initialized resources, and ordered passes.
        unsafe {
            match self {
                Self::Generated(device) => batch
                    .push(super::kernels::add_offsets(
                        device, config, local, offsets, output, n,
                    ))
                    .map(drop),
                Self::Reference(kernels) => batch
                    .push(Launch::new(
                        Ok(&kernels.add_offsets),
                        config,
                        (local, offsets, output, n),
                        |(local, offsets, output, n)| {
                            Ok([
                                Argument::read(local),
                                Argument::read(offsets),
                                Argument::write(output),
                                Argument::value(*n)?,
                            ])
                        },
                    ))
                    .map(drop),
            }
        }
    }
    pub(super) unsafe fn mark(
        &self,
        batch: &mut Batch,
        input: &Buffer<u32>,
        flags: &mut Buffer<u32>,
        n: u32,
    ) -> Result<()> {
        let config = LaunchConfig::<256>::for_elements(n)?;
        // SAFETY: the workload supplies matching bounds, initialized resources, and ordered passes.
        unsafe {
            match self {
                Self::Generated(device) => batch
                    .push(super::kernels::mark(device, config, input, flags, n))
                    .map(drop),
                Self::Reference(kernels) => batch
                    .push(Launch::new(
                        Ok(&kernels.mark),
                        config,
                        (input, flags, n),
                        |(input, flags, n)| {
                            Ok([
                                Argument::read(input),
                                Argument::write(flags),
                                Argument::value(*n)?,
                            ])
                        },
                    ))
                    .map(drop),
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn scatter(
        &self,
        batch: &mut Batch,
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
                Self::Generated(device) => batch
                    .push(super::kernels::scatter(
                        device, config, input, flags, prefix, output, count, n,
                    ))
                    .map(drop),
                Self::Reference(kernels) => batch
                    .push(Launch::new(
                        Ok(&kernels.scatter),
                        config,
                        (input, flags, prefix, output, count, n),
                        |(input, flags, prefix, output, count, n)| {
                            Ok([
                                Argument::read(input),
                                Argument::read(flags),
                                Argument::read(prefix),
                                Argument::write(output),
                                Argument::write(count),
                                Argument::value(*n)?,
                            ])
                        },
                    ))
                    .map(drop),
            }
        }
    }
}
