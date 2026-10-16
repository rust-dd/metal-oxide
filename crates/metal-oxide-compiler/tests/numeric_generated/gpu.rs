use crate::{
    numeric_generated::{SEEDS, atomics, compile, half, wide},
    support,
};
use metal_oxide::{Argument, Device, Launch, LaunchConfig, Operation, Pipeline};

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn generated_half_expression_graphs_match_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    for seed in SEEDS {
        let program = half::Program::generate(seed);
        let directory = compile(&format!("gpu-half-{seed:016x}"), &program.source());
        let module = support::artifact::module(&device, &directory)?;
        let pipeline = Pipeline::new(&device, &module, "generated")?;
        let values = half::inputs(seed);
        let input = device.buffer_from_slice(&values)?;
        let n = values.len() as u32;
        let words = values.len() * program.words();
        let mut output = device.buffer_from_slice(&vec![0xdead_beef_u32; words + 17])?;
        // SAFETY: thread i reads its input and writes its own result words; padded threads return.
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
        let output = output.read().wait();
        for (thread, (&input, output)) in values
            .iter()
            .zip(output[..words].chunks_exact(program.words()))
            .enumerate()
        {
            program.verify(input, output).unwrap_or_else(|error| {
                panic!(
                    "seed {seed:#x}, thread {thread}, {}: {error}",
                    directory.display()
                )
            });
        }
        assert_eq!(&output[words..], &[0xdead_beef; 17]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn generated_float_atomic_state_machines_match_cpu() -> metal_oxide::Result<()> {
    float_atomic_state_machines(false)
}

#[test]
#[ignore = "requires macOS 27 and a Metal device supporting MSL 4.1 atomics"]
fn msl41_generated_float_ordered_atomics_match_cpu() -> metal_oxide::Result<()> {
    float_atomic_state_machines(true)
}

fn float_atomic_state_machines(ordered: bool) -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    for seed in SEEDS {
        let program = atomics::FloatProgram::generate(seed, ordered);
        let directory = compile(
            &format!("gpu-float-{ordered}-{seed:016x}"),
            &program.source(),
        );
        let module = support::artifact::module(&device, &directory)?;
        let pipeline = Pipeline::new(&device, &module, "generated")?;
        let values = atomics::inputs(seed);
        let input = device.buffer_from_slice(&values)?;
        let initial = values
            .iter()
            .map(|input| f32::from_bits(input[0]))
            .collect::<Vec<_>>();
        let mut cell = device.buffer_from_slice(&initial)?;
        let n = values.len() as u32;
        let words = values.len() * program.words();
        let mut output = device.buffer_from_slice(&vec![0xdead_beef_u32; words + 17])?;
        // SAFETY: thread i owns one initialized atomic cell and its output words; padding returns.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<64>::for_elements(n)?,
                (&input, &mut cell, &mut output, n),
                |(input, cell, output, n)| {
                    Ok([
                        Argument::read(input),
                        Argument::atomic(cell),
                        Argument::write(output),
                        Argument::value(*n)?,
                    ])
                },
            )
        }
        .wait()?;
        let output = output.read().wait();
        let cells = cell.read().wait();
        for (thread, (&input, output)) in values
            .iter()
            .zip(output[..words].chunks_exact(program.words()))
            .enumerate()
        {
            program
                .verify(input, output, cells[thread].to_bits())
                .unwrap_or_else(|error| {
                    panic!(
                        "seed {seed:#x}, thread {thread}, {}: {error}",
                        directory.display()
                    )
                });
        }
        assert_eq!(&output[words..], &[0xdead_beef; 17]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn generated_contended_u64_min_max_chains_match_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    for seed in SEEDS {
        let program = wide::Program::generate(seed);
        let directory = compile(&format!("gpu-wide-{seed:016x}"), &program.source());
        let module = support::artifact::module(&device, &directory)?;
        let pipeline = Pipeline::new(&device, &module, "generated")?;
        let values = wide::inputs(seed);
        let input = device.buffer_from_slice(&values)?;
        let mut cells = device.buffer_from_slice(&[u64::MAX, 0])?;
        let n = values.len() as u32;
        // SAFETY: every active thread reads an input and atomically updates two initialized cells.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<256>::for_elements(n)?,
                (&input, &mut cells, n),
                |(input, cells, n)| {
                    Ok([
                        Argument::read(input),
                        Argument::atomic(cells),
                        Argument::value(*n)?,
                    ])
                },
            )
        }
        .wait()?;
        assert_eq!(
            cells.read().wait(),
            program.reference(&values),
            "seed {seed:#x}, {}",
            directory.display()
        );
    }
    Ok(())
}
