#[test]
#[ignore = "requires a Metal device"]
fn kernels_match_the_cpu() -> metal_oxide::Result<()> {
    super::run()
}
