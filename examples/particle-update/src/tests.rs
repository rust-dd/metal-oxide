use super::run;

#[test]
#[ignore = "requires a Metal device"]
fn generated_particle_update_matches_cpu_and_msl() -> metal_oxide::Result<()> {
    run::verify()
}

#[test]
#[ignore = "requires a Metal device"]
fn generated_particle_chain_preserves_nested_fields() -> metal_oxide::Result<()> {
    run::verify_chain()
}

#[test]
#[ignore = "requires a Metal device"]
fn dropped_and_forgotten_generated_particle_operations_wait_for_readback() -> metal_oxide::Result<()>
{
    run::verify_cancelled()
}

#[test]
fn shared_records_keep_the_gpu_layout() {
    use metal_oxide::GpuValue;
    use particle_update::particles::{Config, Motion, Particle};
    assert_eq!((Particle::SIZE, Particle::ALIGNMENT), (28, 4));
    assert_eq!(Motion::SIZE, 16);
    assert_eq!(Config::SIZE, 12);
}
