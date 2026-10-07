#![cfg(feature = "rustc-private")]

mod support;

use std::{path::PathBuf, process::Command};

#[test]
fn generated_record_codecs_work_on_stable_rust() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/structured.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let source = std::fs::read_to_string(directory.join("bindings.rs")).unwrap();
    let records = source.split("pub struct Kernels").next().unwrap();
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
    std::fs::write(project.join("src/main.rs"), format!("{records}\n{}", r#"
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
"#)).unwrap();
    let output = Command::new("cargo")
        .env("RUSTUP_TOOLCHAIN", "stable")
        .current_dir(project)
        .args(["run", "--offline", "--quiet", "--target-dir"])
        .arg(root.join("target/host-codecs"))
        .output()
        .unwrap();
    support::checked(output);
}
