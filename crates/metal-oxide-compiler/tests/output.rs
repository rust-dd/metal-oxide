#![cfg(feature = "rustc-private")]

mod support;

#[test]
fn frontend_errors_invalidate_previous_generated_files() {
    for (body, error) in [
        ("pub unsafe fn broken() {", "unclosed delimiter"),
        (
            "pub unsafe fn broken() { let value: u32 = true; let _ = value; }",
            "E0308",
        ),
        (
            "pub unsafe fn broken() { let mut value = 0u32; let first = &mut value; let second = &mut value; *first = 1; *second = 2; }",
            "E0499",
        ),
    ] {
        let (output, directory) =
            support::emit("crates/metal-oxide-compiler/tests/fixtures/m2_unit.rs", &[]);
        support::checked(output);
        let source = directory.join("invalid.rs");
        std::fs::write(
            &source,
            format!("#![no_std]\nuse metal_oxide_device::kernel;\n#[kernel]\n{body}\n"),
        )
        .unwrap();
        let output = support::emit_into(source.to_str().unwrap(), &[], &directory);
        support::rejected(output, error);
        for name in [
            "kernels.metal",
            "kernels.oxide-ir",
            "abi.json",
            "bindings.rs",
            "rustc-args.json",
        ] {
            assert!(!directory.join(name).exists(), "stale {name} after {error}");
        }
    }
}
