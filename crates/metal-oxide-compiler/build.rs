fn main() {
    let output = std::process::Command::new(std::env::var_os("RUSTC").unwrap())
        .arg("-vV")
        .output()
        .expect("read compiler toolchain identity");
    assert!(output.status.success(), "rustc -vV failed");
    let directory = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    std::fs::write(directory.join("rustc-version"), output.stdout).unwrap();
    println!("cargo:rerun-if-changed=build.rs");
}
