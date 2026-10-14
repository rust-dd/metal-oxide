use super::*;

#[test]
#[ignore = "requires the Metal toolchain"]
fn metal_diagnostics_point_at_rust_lines() {
    let (output, directory) =
        support::emit("crates/metal-oxide-compiler/tests/fixtures/vec_add.rs", &[]);
    support::checked(output);
    let path = directory.join("kernels.metal");
    let msl = std::fs::read_to_string(&path).unwrap();
    let directive = msl.find("#line ").unwrap();
    let end = directive + msl[directive..].find('\n').unwrap() + 1;
    let line = msl[directive + 6..end].split(' ').next().unwrap();
    // A statement Metal rejects, which the directive attributes to that Rust line.
    std::fs::write(
        &path,
        format!(
            "{}    metal_oxide_undeclared;\n{}",
            &msl[..end],
            &msl[end..]
        ),
    )
    .unwrap();
    assert!(metal_oxide_build::compile_msl(&directory).is_err());
    let log = std::fs::read_to_string(directory.join("metal.log")).unwrap();
    assert!(log.contains(&format!("vec_add.rs:{line}:")), "{log}");
}
