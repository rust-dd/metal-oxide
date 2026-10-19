use objc2_metal::{MTLDevice, MTLGPUFamily};

use super::*;

#[cfg(test)]
mod tests;

/// The mpp candidate of `input`, or the reason this machine cannot run it.
pub(super) fn candidate<'a, T: GpuValue>(
    device: &Device,
    config: &Config,
    problem: &'a Problem,
    input: &str,
    values: (&'a [T], &'a [T]),
) -> Result<std::result::Result<Kernel<'a, T>, String>> {
    let path = match &config.gemm_mpp {
        Some(MppLibrary::Metallib(path)) => path,
        Some(MppLibrary::Unavailable(reason)) => return Ok(Err(reason.clone())),
        None => return Err("the gemm workload needs the config's gemm_mpp".into()),
    };
    if let Some(reason) = unsupported(device) {
        return Ok(Err(reason));
    }
    let mut case = kernel_case(config, problem, "mpp", input);
    let entry = format!("gemm_{input}_mpp");
    let loaded = case.load(device, || Module::from_metallib(device, path), &[&entry]);
    let pipeline = match accepted(loaded)? {
        Ok(mut pipelines) => pipelines.remove(0),
        Err(reason) => return Ok(Err(format!("{} rejected it: {reason}", device.name()))),
    };
    // execution_simdgroups<4> is undefined unless the 128 threads of a block form 4 SIMD groups.
    let (name, lanes) = (device.name(), pipeline.thread_execution_width());
    if lanes != 32 {
        return Ok(Err(format!("{name} runs {lanes}-lane SIMD groups")));
    }
    let kernel = Kernel::with_pipeline(device, case, pipeline, problem, values)?;
    Ok(Ok(kernel))
}

/// Why `device` cannot run MSL 4.0 tensor operations, if it cannot.
fn unsupported(device: &Device) -> Option<String> {
    let native = device.as_native();
    let metal4 = objc2::available!(macos = 26.0) && native.supportsFamily(MTLGPUFamily::Metal4);
    (!metal4).then(|| format!("{} lacks the Metal 4 GPU family", device.name()))
}

/// `loaded`, or the reason of a device that rejected the library or its pipeline; other errors
/// stay errors.
fn accepted<P>(loaded: metal_oxide::Result<P>) -> Result<std::result::Result<P, String>> {
    match loaded {
        Ok(loaded) => Ok(Ok(loaded)),
        Err(error @ (metal_oxide::Error::Library(_) | metal_oxide::Error::Pipeline(_))) => {
            Ok(Err(error.to_string()))
        }
        Err(error) => Err(error.into()),
    }
}
