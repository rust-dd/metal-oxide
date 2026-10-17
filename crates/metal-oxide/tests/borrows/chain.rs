use metal_oxide::{Argument, Buffer, Launch, LaunchConfig, Operation, Pipeline};

fn launch<'a>(pipeline: &Pipeline, output: &'a mut Buffer<u32>) -> Launch<(&'a mut Buffer<u32>,), 1> {
    let config = LaunchConfig::<32>::for_elements(32).unwrap();
    // SAFETY: never runs; the borrow conflict stops compilation.
    unsafe { Launch::new(Ok(pipeline), config, (output,), |(output,)| Ok([Argument::write(output)])) }
}

fn check(pipeline: &Pipeline, output: &mut Buffer<u32>) {
    let chain = launch(pipeline, output).map(|(output,)| output.len());
    let values = output.read().wait();
    drop(chain);
    println!("{}", values.len());
}

fn main() {}
