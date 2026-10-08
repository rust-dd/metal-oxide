fn main() {
    println!("cargo:rustc-check-cfg=cfg(metal_oxide_bench)");
    println!("cargo:rerun-if-env-changed=METAL_OXIDE_BENCH_BINDINGS");
    if let Some(path) = std::env::var_os("METAL_OXIDE_BENCH_BINDINGS") {
        println!(
            "cargo:rerun-if-changed={}",
            std::path::Path::new(&path).display()
        );
        let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
        std::fs::copy(path, output.join("bindings.rs")).unwrap();
        println!("cargo:rustc-cfg=metal_oxide_bench");
    }
}
