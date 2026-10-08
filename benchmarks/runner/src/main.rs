#[cfg(metal_oxide_bench)]
mod gpu;
#[cfg(any(test, metal_oxide_bench))]
mod stats;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(metal_oxide_bench)]
    return gpu::run();
    #[cfg(not(metal_oxide_bench))]
    Err("build and run with python3 scripts/bench.py".into())
}
