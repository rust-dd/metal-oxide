use crate::{generated_support, support, vector_generated::programs};
use metal_oxide::{Argument, Device, Launch, LaunchConfig, Operation, Pipeline};

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn generated_vector_graphs_match_host_rust_on_gpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    for program in programs() {
        let name = program.name("gpu");
        let expected = program.cpu(&name);
        let directory = generated_support::compile(&name, &program.source());
        let module = support::artifact::module(&device, &directory)?;
        let input = device.buffer_from_slice(&program.inputs)?;
        let n = program.inputs.len() as u32;
        let words = program.inputs.len() * program.words();
        let mut output = device.buffer_from_slice(&vec![0xdead_beef_u32; words + 17])?;
        for kernel in program.kernels() {
            let pipeline = Pipeline::new(&device, &module, &kernel)?;
            // SAFETY: each active thread writes its graph's words; padded threads return.
            unsafe {
                Launch::new(
                    Ok(&pipeline),
                    LaunchConfig::<64>::for_elements(n)?,
                    (&input, &mut output, n),
                    |(input, output, n)| {
                        Ok([
                            Argument::read(input),
                            Argument::write(output),
                            Argument::value(*n)?,
                        ])
                    },
                )
            }
            .wait()?;
        }
        let output = output.read().wait();
        for (thread, (expected, actual)) in expected
            .iter()
            .zip(output[..words].chunks_exact(program.words()))
            .enumerate()
        {
            if let Some(error) = program.mismatch(expected, actual) {
                panic!("{name}, thread {thread}, {}: {error}", directory.display());
            }
        }
        assert_eq!(&output[words..], &[0xdead_beef; 17]);
    }
    Ok(())
}
