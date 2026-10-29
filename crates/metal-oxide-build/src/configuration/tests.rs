use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "oxide-configuration-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(path.join(".cargo")).unwrap();
        Self(path.canonicalize().unwrap())
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn added_changed_and_removed_configurations_are_stale() {
    let directory = Directory::new();
    let before = Configuration::record(&directory.0, &directory.0, false).unwrap();
    assert!(before.current(&directory.0));
    let path = directory.0.join(".cargo/config.toml");
    let credential = "secret-configuration-value-never-recorded-as-text";
    std::fs::write(
        &path,
        format!("# {credential}\n[profile.release]\noverflow-checks=true\n"),
    )
    .unwrap();
    assert!(!before.current(&directory.0));
    let present = Configuration::record(&directory.0, &directory.0, false).unwrap();
    assert!(present.current(&directory.0));
    assert!(present.paths(&directory.0).any(|tracked| tracked == path));
    let json = serde_json::to_string(&present).unwrap();
    assert!(!json.contains(credential));
    assert!(!json.contains(directory.0.to_str().unwrap()));
    if let Some(home) = std::env::var_os("HOME") {
        assert!(!json.contains(home.to_str().unwrap()));
    }
    std::fs::write(&path, "[profile.release]\noverflow-checks=false\n").unwrap();
    assert!(!present.current(&directory.0));
    std::fs::remove_file(path).unwrap();
    assert!(!present.current(&directory.0));
    assert!(before.current(&directory.0));
}

#[test]
fn sibling_kernel_workspaces_resolve_the_same_configuration_paths() {
    let directory = Directory::new();
    let host = directory.0.join("host");
    let kernels = directory.0.join("kernels");
    std::fs::create_dir(&host).unwrap();
    std::fs::create_dir(&kernels).unwrap();
    let configuration = Configuration::record(&kernels, &host, false).unwrap();
    assert!(configuration.current(&host));
}

#[test]
fn copied_workspaces_ignore_their_generated_target_configuration() {
    let directory = Directory::new();
    let generated = directory.0.join("generated");
    std::fs::create_dir_all(generated.join(".cargo")).unwrap();
    let configuration = Configuration::record(&generated, &directory.0, true).unwrap();
    std::fs::write(
        generated.join(".cargo/config.toml"),
        "[build]\ntarget-dir='elsewhere'\n",
    )
    .unwrap();
    assert!(configuration.current(&directory.0));
    // Copies of the same kernels at other keys record the same configuration.
    let other = directory.0.join("other");
    std::fs::create_dir(&other).unwrap();
    assert_eq!(
        Configuration::record(&other, &directory.0, true).unwrap(),
        configuration
    );
    std::fs::write(
        directory.0.join(".cargo/config.toml"),
        "[profile.release]\noverflow-checks=true\n",
    )
    .unwrap();
    assert!(!configuration.current(&directory.0));
}

#[test]
fn identical_sources_and_configuration_remain_current_at_another_directory_depth() {
    let directory = Directory::new();
    let original = directory.0.join("original");
    let relocated = directory.0.join("another/deeper/location");
    for root in [&original, &relocated] {
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir(root.join(".cargo")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn value() -> u32 { 1 }\n").unwrap();
        std::fs::write(
            root.join(".cargo/config.toml"),
            "[profile.release]\noverflow-checks=true\n",
        )
        .unwrap();
    }
    let configuration = Configuration::record(&original, &original, false).unwrap();
    assert!(configuration.current(&relocated));
    let ancestor = directory.0.join("another/.cargo");
    std::fs::create_dir(&ancestor).unwrap();
    let path = ancestor.join("config.toml");
    std::fs::write(
        &path,
        "[build]\nrustflags=['-C', 'overflow-checks=false']\n",
    )
    .unwrap();
    assert!(!configuration.current(&relocated));
    std::fs::remove_file(path).unwrap();
    assert!(configuration.current(&relocated));
}

#[test]
fn cargo_transport_credentials_aliases_and_output_do_not_change_kernel_settings() {
    let directory = Directory::new();
    let before = Configuration::record(&directory.0, &directory.0, false).unwrap();
    let path = directory.0.join(".cargo/config.toml");
    std::fs::write(&path, "[http]\nmultiplexing=false\n[net]\ngit-fetch-with-cli=true\n[registry]\ntoken='secret'\n[alias]\nb='build'\n[term]\ncolor='always'\n[build]\ntarget-dir='output'\njobs=2\n[profile.release]\n[env]\n[target]\n").unwrap();
    assert!(before.current(&directory.0));
    let after = Configuration::record(&directory.0, &directory.0, false).unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(after.current(&directory.0));
}

#[test]
fn formatting_and_comments_do_not_change_kernel_settings() {
    let directory = Directory::new();
    let path = directory.0.join(".cargo/config.toml");
    std::fs::write(&path, "[profile.release]\noverflow-checks=true\n").unwrap();
    let before = Configuration::record(&directory.0, &directory.0, false).unwrap();
    std::fs::write(
        &path,
        "# private comment\n[profile . release]\noverflow-checks = true # unchanged\n",
    )
    .unwrap();
    assert!(before.current(&directory.0));
}

#[test]
fn semantic_build_target_environment_and_profile_settings_are_retained() {
    let directory = Directory::new();
    let path = directory.0.join(".cargo/config.toml");
    let before = Configuration::record(&directory.0, &directory.0, false).unwrap();
    for settings in [
        "[build]\nrustflags=['-C', 'overflow-checks=true']\n",
        "[target.'cfg(target_env = \"metal\")']\nrustflags=['--cfg', 'kernel_probe']\n",
        "[env]\nKERNEL_STEP={value='2',force=true}\n",
        "[profile.release]\noverflow-checks=true\n",
    ] {
        std::fs::write(&path, settings).unwrap();
        assert!(!before.current(&directory.0), "ignored {settings}");
        let present = Configuration::record(&directory.0, &directory.0, false).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(
            !present.current(&directory.0),
            "ignored removal of {settings}"
        );
    }
}

#[test]
fn config_includes_are_rejected_instead_of_missing_transitive_changes() {
    let directory = Directory::new();
    std::fs::write(
        directory.0.join(".cargo/config.toml"),
        "include=['profile.toml']\n",
    )
    .unwrap();
    std::fs::write(
        directory.0.join(".cargo/profile.toml"),
        "[profile.release]\noverflow-checks=true\n",
    )
    .unwrap();
    let error = Configuration::record(&directory.0, &directory.0, false)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("configuration includes are unsupported"),
        "{error}"
    );
    assert!(error.contains("config.toml"), "{error}");
}

#[test]
fn cargo_home_configuration_is_not_repeated_in_ancestor_files() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let configuration = Configuration::record(root, root, false).unwrap();
    if let Some(home) = cargo_home(root) {
        assert!(!configuration.files.keys().any(|path| {
            let path = normalize(&root.join(path));
            path == home.join("config") || path == home.join("config.toml")
        }));
    }
}

#[test]
fn forced_environment_uses_the_nearest_explicit_override() {
    let directory = Directory::new();
    let root = directory.0.join("nested");
    std::fs::create_dir_all(root.join(".cargo")).unwrap();
    std::fs::write(
        directory.0.join(".cargo/config.toml"),
        "[env]\nFIRST={value='outer',force=true}\nSECOND={value='outer',force=true}\n",
    )
    .unwrap();
    std::fs::write(
        root.join(".cargo/config.toml"),
        "[env]\nFIRST={value='inner',force=false}\nSECOND={value='inner'}\nTHIRD='default'\n",
    )
    .unwrap();
    let configuration = Configuration::record(&root, &root, false).unwrap();
    assert_eq!(
        configuration.environment,
        BTreeMap::from([
            ("FIRST".into(), false),
            ("SECOND".into(), true),
            ("THIRD".into(), false),
        ])
    );
}
