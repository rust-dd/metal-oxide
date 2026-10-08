fn main() {
    println!("cargo:rerun-if-env-changed=METAL_OXIDE_BINDINGS");
    println!("cargo:rerun-if-env-changed=METAL_OXIDE_ARTIFACT_DIR");
    let bindings = std::env::var("METAL_OXIDE_BINDINGS").expect("build with cargo metal");
    println!("cargo:rerun-if-changed={bindings}");
}
