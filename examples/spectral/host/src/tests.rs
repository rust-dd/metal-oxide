use super::*;

#[test]
#[ignore = "requires a Metal device"]
fn generated_spectral_kernels_match_cpu_and_native() {
    validation::run().unwrap();
}
