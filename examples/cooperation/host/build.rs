fn main() {
    println!("cargo:rustc-check-cfg=cfg(metal_oxide_artifact)");
    println!("cargo:rerun-if-env-changed=METAL_OXIDE_BINDINGS");
    println!("cargo:rerun-if-env-changed=METAL_OXIDE_ARTIFACT_DIR");
    if let Some(bindings) = std::env::var_os("METAL_OXIDE_BINDINGS") {
        println!(
            "cargo:rerun-if-changed={}",
            std::path::Path::new(&bindings).display()
        );
        println!("cargo:rustc-cfg=metal_oxide_artifact");
    }
}
