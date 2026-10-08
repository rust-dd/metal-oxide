#![cfg(feature = "rustc-private")]

mod support;

use std::{path::PathBuf, process::Command};

#[test]
fn generated_record_codecs_work_on_stable_rust() {
    run_codecs(
        "crates/metal-oxide-compiler/tests/fixtures/structured.rs",
        r#"
use metal_oxide::GpuValue;
fn main() {
    assert_eq!((Particle::SIZE, Particle::ALIGNMENT), (24, 4));
    assert_eq!(Config::SIZE, 12);
    let value = Particle { tag: 7, flags: 0x0302, id: 0x0706_0504, position: (1.0, 2.0), velocity: [3.0, 4.0] };
    let mut bytes = [0xff; 24];
    value.encode(&mut bytes);
    assert_eq!(bytes, [7, 0, 2, 3, 4, 5, 6, 7, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64]);
    assert_eq!(Particle::decode(&bytes), value);
    bytes[1] = 0xff;
    assert_eq!(Particle::decode(&bytes), value);
    assert_eq!(Particle::zeroed().position, (0.0, 0.0));
    let mut array = [0xff; 48];
    [value; 2].encode(&mut array);
    assert_eq!(<[Particle; 2]>::decode(&array), [value; 2]);
    assert_eq!(&array[0..24], &array[24..48]);
}
"#,
    );
}

#[test]
fn generated_record_names_do_not_shadow_binding_dependencies() {
    run_codecs(
        "crates/metal-oxide-compiler/tests/fixtures/host_names.rs",
        r#"
use metal_oxide::GpuValue;
fn main() {
    let value = usize2 { value: 7 };
    let mut bytes = [0; 4];
    value.encode(&mut bytes);
    assert_eq!(usize2::decode(&bytes), value);
    assert_eq!(AsRef::SIZE, 4);
    assert_eq!(Ok::SIZE, 4);
    assert_eq!(Into::SIZE, 4);
}
"#,
    );
}

#[test]
fn generated_signed_record_codecs_preserve_bits_and_padding() {
    run_codecs(
        "crates/metal-oxide-compiler/tests/fixtures/signed_narrow.rs",
        r#"
use metal_oxide::GpuValue;
fn main() {
    assert_eq!((SignedRecord::SIZE, SignedRecord::ALIGNMENT), (8, 2));
    let value = SignedRecord { small: -128, wide: -32768, pair: (-1, 0x1234) };
    let mut bytes = [0xff; 8];
    value.encode(&mut bytes);
    assert_eq!(bytes, [128, 0, 0, 128, 255, 0, 52, 18]);
    assert_eq!(SignedRecord::decode(&bytes), value);
    bytes[1] = 0xaa;
    bytes[5] = 0xbb;
    assert_eq!(SignedRecord::decode(&bytes), value);
}
"#,
    );
}

#[test]
fn generated_half_record_codecs_work_on_stable_rust() {
    run_codecs(
        "crates/metal-oxide-compiler/tests/fixtures/half.rs",
        r#"
use metal_oxide::{F16, GpuValue};
fn main() {
    assert_eq!((HalfRecord::SIZE, HalfRecord::ALIGNMENT), (10, 2));
    let value = HalfRecord { tag: 7, pair: (F16::from_bits(0x3c00), F16::from_bits(0xbc00)), samples: [F16::from_bits(1), F16::INFINITY] };
    let mut bytes = [0xff; 10];
    value.encode(&mut bytes);
    assert_eq!(bytes, [7, 0, 0, 60, 0, 188, 1, 0, 0, 124]);
    bytes[1] = 0xaa;
    assert_eq!(HalfRecord::decode(&bytes), value);
    let layout = HalfRecord::layout().unwrap();
    layout.validate().unwrap();
    assert_eq!((layout.size, layout.alignment), (10, 2));
}
"#,
    );
}

fn run_codecs(fixture: &str, program: &str) {
    let (output, directory) = support::emit(fixture, &["-C", "overflow-checks=off"]);
    support::checked(output);
    let source = std::fs::read_to_string(directory.join("bindings.rs")).unwrap();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let project = directory.join("host-codecs");
    std::fs::create_dir_all(project.join("src")).unwrap();
    std::fs::write(project.join("Cargo.toml"), format!(
        "[workspace]\n[workspace.dependencies]\nmetal-oxide = {{ path = {:?} }}\n[package]\nname = \"host-codecs\"\nversion = \"0.0.0\"\nedition = \"2024\"\n[dependencies]\nmetal-oxide.workspace = true\n",
        root.join("crates/metal-oxide").to_str().unwrap()
    )).unwrap();
    std::fs::write(
        project.join("src/main.rs"),
        format!("#![allow(non_camel_case_types)]\n{source}\n{program}"),
    )
    .unwrap();
    std::fs::copy(
        root.join("rust-toolchain.toml"),
        project.join("rust-toolchain.toml"),
    )
    .unwrap();
    let output = Command::new("cargo")
        .env_remove("RUSTUP_TOOLCHAIN")
        .current_dir(project)
        .args(["run", "--offline", "--quiet", "--target-dir"])
        .arg(root.join("target/host-codecs"))
        .output()
        .unwrap();
    support::checked(output);
}
