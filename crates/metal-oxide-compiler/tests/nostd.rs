#![cfg(feature = "rustc-private")]

mod support;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "nostd/gpu.rs"]
mod gpu;
#[path = "nostd/msl.rs"]
mod msl;

use std::{path::PathBuf, process::Command, sync::OnceLock};

fn cargo_kernel(feature: Option<&str>) -> (PathBuf, Command) {
    let root = support::root().canonicalize().unwrap();
    let directory = root.join(format!("target/nostd-tests/{}", std::process::id()));
    let output = directory.join(feature.unwrap_or("output"));
    let target = root.join("crates/metal-oxide-build/targets/metal64-unknown-none.json");
    let mut cargo = Command::new("cargo");
    cargo
        .current_dir(root)
        .args([
            "check",
            "-Zbuild-std=core",
            "-Zjson-target-spec",
            "--release",
            "--lib",
            "--locked",
            "--package",
            "nostd-numeric-kernels",
        ])
        .arg("--target")
        .arg(&target)
        .arg("--target-dir")
        .arg(directory)
        .env("RUSTC_WRAPPER", env!("CARGO_BIN_EXE_metal-oxide-compiler"))
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env("RUSTFLAGS", "-Zunstable-options")
        .env("METAL_OXIDE_CARGO_TARGET", target)
        .env("METAL_OXIDE_CARGO_KERNEL", "nostd_numeric_kernels")
        .env("METAL_OXIDE_CARGO_OUTPUT", &output);
    if let Some(feature) = feature {
        cargo.args(["--features", feature]);
    }
    (output, cargo)
}

fn kernels() -> &'static std::path::Path {
    static OUTPUT: OnceLock<PathBuf> = OnceLock::new();
    OUTPUT.get_or_init(|| {
        let (output, mut cargo) = cargo_kernel(None);
        support::checked(cargo.output().unwrap());
        output
    })
}

#[test]
fn cargo_imports_libm_and_num_traits_mir() {
    let output = kernels();
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(output.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(abi.kernels.len(), 9);
    assert!(
        abi.kernels
            .iter()
            .all(|kernel| kernel.safe && kernel.one_dimensional && kernel.lengths)
    );
    let msl = std::fs::read_to_string(output.join("kernels.metal")).unwrap();
    for implementation in [
        "libm-0.2.16/src/math/modff.rs",
        "libm-0.2.16/src/math/generic/floor.rs",
        "num-traits-0.2.19/src/float.rs",
        "num-traits-0.2.19/src/pow.rs",
    ] {
        assert!(msl.contains(implementation), "missing {implementation}");
    }
}

#[test]
fn libm_f32_cube_root_retains_unsupported_f64_diagnostics() {
    let (output, mut cargo) = cargo_kernel(Some("unsupported-f64"));
    support::rejected(cargo.output().unwrap(), "unsupported device type: f64");
    assert!(!output.join("kernels.metal").exists());
}

#[test]
fn libm_sine_rejects_unsupported_argument_reduction() {
    let (output, mut cargo) = cargo_kernel(Some("unsupported-sine"));
    support::rejected(cargo.output().unwrap(), "volatile_load");
    assert!(!output.join("kernels.metal").exists());
}

fn float_inputs() -> Vec<f32> {
    let mut values = vec![
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.75,
        -2.75,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0xffc0_1234),
    ];
    values.extend((0..4096_u32).map(|i| f32::from_bits(i.wrapping_mul(0x9e37_79b9))));
    values
}

fn assert_float(actual: f32, expected: f32, label: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{label}: {actual:?} != NaN");
    } else {
        assert_eq!(actual.to_bits(), expected.to_bits(), "{label}");
    }
}
