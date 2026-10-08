fn main() {
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let profile = output
        .ancestors()
        .find(|path| path.file_name().is_some_and(|name| name == "build"))
        .and_then(std::path::Path::parent)
        .expect("Cargo profile output directory");
    println!(
        "cargo:rustc-env=METAL_OXIDE_DEV_BIN_DIR={}",
        profile.display()
    );
    println!("cargo:rerun-if-changed=build.rs");
}
