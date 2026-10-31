use super::{
    differential::{Case, Gpu, Outcome},
    generate::model::{Bits, COUNTERS},
    output::POISON,
};
use metal_oxide::{Argument, Buffer, Device, GpuAtomic, GpuValue, Launch, LaunchConfig, Operation};
use std::path::Path;

/// A buffer element type of generated kernels.
trait Element: GpuValue {
    /// `input as Self`, which the CPU passes to `run`.
    fn from_input(input: u32) -> Self;
}

macro_rules! elements {
    ($($ty:ty),*) => {$(
        impl Element for $ty {
            fn from_input(input: u32) -> Self {
                input as Self
            }
        }
    )*};
}

elements!(u8, u16, u32, u64, i8, i16, i32, i64, f32);

/// The value of POISON's repeated bytes, which untouched output elements hold.
fn poison<T: GpuValue>() -> T {
    let bytes = (u64::from(POISON) * 0x1_0000_0001).to_le_bytes();
    T::decode(&bytes[..T::SIZE])
}

/// Appends the words that the CPU prints for `value`, low word first.
fn words<T: GpuValue>(value: T, words: &mut Vec<u32>) {
    let mut bytes = [0; 8];
    value.encode(&mut bytes[..T::SIZE]);
    let bits = u64::from_le_bytes(bytes);
    words.push(bits as u32);
    if T::SIZE == 8 {
        words.push((bits >> 32) as u32);
    }
}

/// Evaluates `body` with `$element` naming the buffer element type `$name`.
macro_rules! element {
    ($name:expr, $element:ident => $body:expr) => {
        match $name {
            "u8" => {
                type $element = u8;
                $body
            }
            "u16" => {
                type $element = u16;
                $body
            }
            "u32" => {
                type $element = u32;
                $body
            }
            "u64" => {
                type $element = u64;
                $body
            }
            "i8" => {
                type $element = i8;
                $body
            }
            "i16" => {
                type $element = i16;
                $body
            }
            "i32" => {
                type $element = i32;
                $body
            }
            "i64" => {
                type $element = i64;
                $body
            }
            "f32" => {
                type $element = f32;
                $body
            }
            other => Err(format!("unknown buffer element type {other}")),
        }
    };
}

impl Case {
    /// What the emitted kernels compute on `device`.
    pub fn gpu(&self, device: &Device, generated: &Path) -> Result<Gpu, String> {
        element!(self.input.as_str(), I => element!(self.output.as_str(), O => {
            self.typed::<I, O>(device, generated)
        }))
    }

    fn typed<I: Element, O: Element>(
        &self,
        device: &Device,
        generated: &Path,
    ) -> Result<Gpu, String> {
        use metal_oxide::Pipeline;
        let module = super::artifact::module(device, generated).map_err(|e| e.to_string())?;
        let pipeline = Pipeline::new(device, &module, "generated").map_err(|e| e.to_string())?;
        let inputs = self
            .inputs
            .iter()
            .map(|&input| I::from_input(input))
            .collect::<Vec<_>>();
        let input = device
            .buffer_from_slice(&inputs)
            .map_err(|e| e.to_string())?;
        let initial = vec![poison::<O>(); self.layout().length()];
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
                // writes the elements at i * elements.
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
        let mut written = Vec::new();
        for &element in output.read().wait().iter() {
            words(element, &mut written);
        }
        Ok(Gpu {
            failure,
            words: written,
            counters,
        })
    }
}

/// A launch of a generated kernel before its counters, if any, are bound.
struct Launched<'a, I: GpuValue, O: GpuValue> {
    device: &'a Device,
    pipeline: &'a metal_oxide::Pipeline,
    config: LaunchConfig<64>,
    input: &'a Buffer<I>,
    output: &'a mut Buffer<O>,
    n: u32,
}

impl<I: GpuValue, O: GpuValue> Launched<'_, I, O> {
    /// The failed check, if any, and the final counters of a run whose counters of `C` start at
    /// [`COUNTERS`].
    fn counting<C: GpuAtomic + Bits>(self) -> Result<(Option<Outcome>, Vec<u32>), String> {
        let initial = COUNTERS.map(|word| C::from_word(word.into()));
        let mut counters = self
            .device
            .buffer_from_slice(&initial)
            .map_err(|e| e.to_string())?;
        // SAFETY: the buffers are disjoint, padded threads are guarded, thread i alone writes the
        // elements at i * elements, and every access to the counters is atomic.
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
        let words = words
            .map(|&counter| counter.into_word() as u32)
            .collect::<Vec<_>>();
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
