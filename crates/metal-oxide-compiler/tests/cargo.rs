mod support;

use std::{path::PathBuf, process::Command};

fn cargo_kernel(package: &str, crate_name: &str) -> (PathBuf, Command) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let directory = root.join(format!(
        "target/cargo-tests/{}/{package}",
        std::process::id()
    ));
    let output = directory.join("output");
    let mut cargo = Command::new("cargo");
    cargo
        .current_dir(&root)
        .args([
            "check",
            "-Zbuild-std=core",
            "-Zjson-target-spec",
            "--release",
            "--lib",
            "--locked",
            "--package",
            package,
        ])
        .arg("--target")
        .arg(root.join("targets/metal64-unknown-none.json"))
        .arg("--target-dir")
        .arg(&directory)
        .env("RUSTC_WRAPPER", env!("CARGO_BIN_EXE_metal-oxide-compiler"))
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env("RUSTFLAGS", "-Zunstable-options")
        .env(
            "METAL_OXIDE_CARGO_TARGET",
            root.join("targets/metal64-unknown-none.json"),
        )
        .env("METAL_OXIDE_CARGO_KERNEL", crate_name)
        .env("METAL_OXIDE_CARGO_OUTPUT", &output);
    (output, cargo)
}

#[test]
fn cargo_builds_device_core_and_host_macros_and_emits_named_bindings() {
    let (output, mut cargo) = cargo_kernel("vec-add-kernels", "vec_add_kernels");
    support::checked(cargo.output().unwrap());
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(output.join("abi.json")).unwrap(),
    )
    .unwrap();
    let kernel = &abi.kernels[0];
    assert_eq!(kernel.name, "vec_add");
    assert_eq!(
        kernel
            .parameters
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        ["a", "b", "out", "n"]
    );
    let bindings = std::fs::read_to_string(output.join("bindings.rs")).unwrap();
    assert!(bindings.contains("r#out: &mut metal_oxide::Buffer<f32>"));
    assert!(bindings.contains("r#n: u32"));
    assert!(
        std::fs::read_to_string(output.join("kernels.metal"))
            .unwrap()
            .contains("kernel void vec_add")
    );

    cargo.env("RUSTFLAGS", "-Zunstable-options -Coverflow-checks=on");
    support::rejected(cargo.output().unwrap(), "MIR assertion");
    for name in ["kernels.metal", "abi.json", "bindings.rs"] {
        assert!(
            !output.join(name).exists(),
            "stale {name} after rejected rebuild"
        );
    }
}

#[test]
fn cross_crate_nested_records_and_const_helpers_lower_through_cargo() {
    let (directory, mut cargo) = cargo_kernel("particle-update-kernels", "particle_update_kernels");
    support::checked(cargo.output().unwrap());
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(abi.kernels.len(), 2);
    assert!(
        abi.kernels
            .iter()
            .all(|kernel| kernel.required_block == Some([64, 1, 1]))
    );
    let parameter = &abi.kernels[0].parameters[0].ty;
    let metal_oxide_artifact::ParameterType::Buffer {
        element, stride, ..
    } = parameter
    else {
        panic!()
    };
    assert_eq!((*stride, element.size, element.alignment), (28, 28, 4));
    let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    let signature = msl
        .lines()
        .find(|line| line.starts_with("kernel void particle_update("))
        .unwrap();
    let particle = signature
        .split("device const ")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap();
    let config = signature
        .split("constant ")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap();
    let program = format!(
        "{particle} input = {{uchar(1),ushort(3),9u,{{{{1.0f,2.0f}},{{{{3.0f,4.0f}}}}}},{{{{uchar(5),uchar(6),uchar(7)}}}}}}; {particle} output = {{}}; {config} config = {{0.5f,{{1.0f,2.0f}}}}; uint n = 1; particle_update(&input, &output, config, n, {{}}, {{}}, {{64,1,1}}, {{1,1,1}}); std::cout << output.f0 + 0 << ',' << output.f1 << ',' << output.f2 << ',' << output.f3.f0.f0 << ',' << output.f3.f0.f1 << ',' << output.f3.f1.elements[0] << ',' << output.f3.f1.elements[1] << ',' << output.f4.elements[1] + 0;"
    );
    assert_eq!(
        support::execute_msl(&directory, &program),
        "1,5,9,4.75,7.5,4,6,5"
    );
}

#[test]
fn cargo_imports_external_struct_helpers_and_const_specializations() {
    let (output, mut cargo) = cargo_kernel("pipeline-kernels", "pipeline_kernels");
    support::checked(cargo.output().unwrap());
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(output.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(abi.kernels.len(), 3);
    for (name, block) in [("scale_128", [128, 1, 1]), ("scale_256", [256, 1, 1])] {
        let kernel = abi
            .kernels
            .iter()
            .find(|kernel| kernel.name == name)
            .unwrap();
        assert_eq!(kernel.required_block, Some(block));
    }
    let msl = std::fs::read_to_string(output.join("kernels.metal")).unwrap();
    assert!(msl.contains("metal_oxide_aggregate_"));
    assert!(msl.contains("kernel void scale_128"));
    assert!(msl.contains("kernel void scale_256"));
    let bindings = std::fs::read_to_string(output.join("bindings.rs")).unwrap();
    assert!(bindings.contains("fn r#enqueue_scale_128"));
    assert!(bindings.contains("config: metal_oxide::LaunchConfig<128, 1, 1>"));
    assert!(bindings.contains("config: metal_oxide::LaunchConfig<256, 1, 1>"));
}
