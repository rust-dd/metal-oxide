use super::{
    differential::{Case, Gpu, Outcome},
    generate::model::{Bits, COUNTERS},
};
use std::path::Path;

impl Case {
    /// What the emitted kernels compute on `device`.
    pub fn gpu(&self, device: &metal_oxide::Device, generated: &Path) -> Result<Gpu, String> {
        use metal_oxide::{Argument, Launch, LaunchConfig, Operation, Pipeline};
        let module = super::artifact::module(device, generated).map_err(|e| e.to_string())?;
        let pipeline = Pipeline::new(device, &module, "generated").map_err(|e| e.to_string())?;
        let input = device
            .buffer_from_slice(&self.inputs)
            .map_err(|e| e.to_string())?;
        let initial = super::output::initial(self.inputs.len(), self.words.len());
        let mut output = device
            .buffer_from_slice(&initial)
            .map_err(|e| e.to_string())?;
        let n = self.inputs.len() as u32;
        let config = LaunchConfig::<64>::for_elements(n).map_err(|e| e.to_string())?;
        let launch = Launched {
            device,
            pipeline: &pipeline,
            config,
            input: &input,
            output: &mut output,
            n,
        };
        let (failure, counters) = match self.counters.as_deref() {
            None => {
                // SAFETY: the buffers are disjoint; padded threads are guarded and thread i alone
                // writes the words at i * words.
                let result = unsafe {
                    Launch::new(
                        Ok(launch.pipeline),
                        launch.config,
                        (launch.input, launch.output, n),
                        |(input, output, n)| {
                            Ok([
                                Argument::read(input),
                                Argument::write(output),
                                Argument::value(*n)?,
                            ])
                        },
                    )
                }
                .wait();
                (failure(result)?, Vec::new())
            }
            Some("u32") => launch.counting::<u32>()?,
            Some("i32") => launch.counting::<i32>()?,
            Some(other) => return Err(format!("unknown counter type {other}")),
        };
        Ok(Gpu {
            failure,
            words: output.read().wait().to_vec(),
            counters,
        })
    }
}

/// A launch of a generated kernel before its counters, if any, are bound.
struct Launched<'a> {
    device: &'a metal_oxide::Device,
    pipeline: &'a metal_oxide::Pipeline,
    config: metal_oxide::LaunchConfig<64>,
    input: &'a metal_oxide::Buffer<u32>,
    output: &'a mut metal_oxide::Buffer<u32>,
    n: u32,
}

impl Launched<'_> {
    /// The failed check, if any, and the final counters of a run whose counters of `C` start at
    /// [`COUNTERS`].
    fn counting<C: metal_oxide::GpuAtomic + Bits>(
        self,
    ) -> Result<(Option<Outcome>, Vec<u32>), String> {
        use metal_oxide::{Argument, Launch, Operation};
        let initial = COUNTERS.map(|word| C::from_word(word.into()));
        let mut counters = self
            .device
            .buffer_from_slice(&initial)
            .map_err(|e| e.to_string())?;
        // SAFETY: the buffers are disjoint, padded threads are guarded, thread i alone writes the
        // words at i * words, and every access to the counters is atomic.
        let result = unsafe {
            Launch::new(
                Ok(self.pipeline),
                self.config,
                (self.input, self.output, self.n, &mut counters),
                |(input, output, n, counters)| {
                    Ok([
                        Argument::read(input),
                        Argument::write(output),
                        Argument::value(*n)?,
                        Argument::atomic(counters),
                    ])
                },
            )
        }
        .wait();
        let failure = failure(result)?;
        let words = counters.read().wait().iter();
        let words = words.map(|&counter| counter.into_word() as u32).collect();
        Ok((failure, words))
    }
}

/// The check that a run failed, if it failed one.
fn failure<T>(result: Result<T, metal_oxide::Error>) -> Result<Option<Outcome>, String> {
    match result {
        Ok(_) => Ok(None),
        Err(metal_oxide::Error::KernelFailed(failure)) => Ok(Some(Outcome::Panic {
            line: failure.line,
            column: failure.column,
        })),
        Err(error) => Err(error.to_string()),
    }
}
