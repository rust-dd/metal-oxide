use super::{SourceKernels, blank};
use crate::metadata::Metadata;
use metal_oxide_artifact::MathMode;
use serde_json::{Value, json};

#[test]
fn blanking_keeps_the_kept_range_at_its_lines_and_columns() {
    let text = "use a::b;\n#[kernel] fn f(x: u32) {\n    é(x)\n}\nfn main() {}\n";
    assert_eq!(
        blank(text, ((2, 11), (4, 2)), None),
        "\n          fn f(x: u32) {\n    é(x)\n}\n"
    );
    let edited = "use a::c;\n#[kernel] fn f(x: u32) {\n    é(x)\n}\nfn other() {}\n";
    assert_eq!(
        blank(edited, ((2, 11), (4, 2)), None),
        blank(text, ((2, 11), (4, 2)), None)
    );
    let module = "#[a] #[kernels] mod m {}\n";
    assert_eq!(
        blank(module, ((1, 1), (1, 25)), Some(((1, 6), (1, 16)))),
        "#[a]            mod m {}\n"
    );
}

fn package(name: &str, manifest: &str, dependencies: Value) -> Value {
    json!({
        "id": format!("{name} 0.1.0"),
        "name": name,
        "version": "0.1.0",
        "edition": "2024",
        "manifest_path": manifest,
        "targets": [],
        "dependencies": dependencies,
    })
}

fn dependency(
    name: &str,
    rename: Option<&str>,
    kind: Option<&str>,
    defaults: bool,
    features: &[&str],
) -> Value {
    json!({
        "name": name,
        "rename": rename,
        "kind": kind,
        "uses_default_features": defaults,
        "features": features,
    })
}

#[test]
fn device_manifests_declare_dependencies_as_their_package_does() {
    let host = serde_json::from_value::<Metadata>(json!({
        "workspace_root": "/w",
        "target_directory": "/w/target",
        "packages": [
            package("host", "/w/Cargo.toml", json!([
                dependency("helper", Some("kernel-helper"), None, false, &["scale"]),
                dependency("helper", Some("kernel-helper"), Some("dev"), true, &["std"]),
                dependency("tested", None, Some("dev"), true, &[]),
            ])),
            package("helper", "/w/helper/Cargo.toml", json!([])),
            package("tested", "/registry/tested/Cargo.toml", json!([])),
            package("metal-oxide-device", "/registry/device/Cargo.toml", json!([])),
        ],
    }))
    .unwrap();
    let mut kernels = SourceKernels {
        package: "/w".into(),
        file: "src/main.rs".into(),
        start: (1, 1),
        end: (1, 1),
        hole: None,
        item: "scale".into(),
        prelude: false,
        math_mode: MathMode::Precise,
        dependencies: vec!["kernel_helper".into(), "tested".into()],
        debug_info: false,
    };
    let manifest = kernels.manifest(&host, "key").unwrap();
    for declaration in [
        "kernel_helper = { package = \"helper\", path = \"/w/helper\", default-features = false, features = [\"scale\"] }\n",
        "tested = { package = \"tested\", path = \"/registry/tested\", default-features = true, features = [] }\n",
    ] {
        assert!(manifest.contains(declaration), "{manifest}");
    }
    kernels.dependencies = vec!["helper".into()];
    let error = kernels.manifest(&host, "key").unwrap_err().to_string();
    assert!(error.contains("host does not declare"), "{error}");
}
