#![cfg(feature = "rustc-private")]

mod support;

use std::{process::Command, time::Duration};

#[test]
fn differential_process_deadline_stops_nonterminating_programs() {
    let error = support::process::output(
        Command::new("sh").args(["-c", "while :; do :; done"]),
        Duration::from_millis(100),
    )
    .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    let output = support::process::output(
        Command::new("sh").args(["-c", "printf complete"]),
        Duration::from_secs(2),
    )
    .unwrap();
    assert_eq!(support::checked(output), "complete");
}

#[test]
fn seeded_rust_matches_cpu_and_emitted_msl() {
    for seed in support::differential::SEEDS {
        let case = support::differential::Case::new(seed);
        let generated = case.compile();
        assert_eq!(
            case.msl(&generated),
            case.cpu(),
            "seed {seed}, source {}",
            case.source.display()
        );
        let original = metal_oxide_artifact::COMPILER_OUTPUTS
            .map(|file| std::fs::read(generated.join(file.name())).unwrap());
        let rebuilt = case.compile();
        for (file, expected) in metal_oxide_artifact::COMPILER_OUTPUTS
            .into_iter()
            .zip(original)
        {
            assert_eq!(
                std::fs::read(rebuilt.join(file.name())).unwrap(),
                expected,
                "nondeterministic {} for seed {seed}",
                file.name()
            );
        }
    }
}

#[test]
fn unsupported_generated_sources_keep_locations_and_never_panic() {
    let case = support::differential::Case::new(42);
    for (name, body, diagnostic) in [
        (
            "division",
            "let x = 9 / n; output.store_unchecked(0, x);",
            "assertion",
        ),
        (
            "pointer",
            "let address = n as *const u32; output.store_unchecked(0, *address);",
            "unsupported",
        ),
        (
            "index",
            "let values = [n; 4]; output.store_unchecked(0, values[n as usize]);",
            "assertion",
        ),
    ] {
        let source = case.directory.join(format!("{name}.rs"));
        std::fs::write(&source, format!("#![no_std]\nuse metal_oxide_device::{{kernel, WriteBuffer}};\n#[kernel]\npub unsafe fn invalid(output: WriteBuffer<u32>, n: u32) {{\nunsafe {{ {body} }}\n}}\n")).unwrap();
        let (output, _) = support::emit(source.to_str().unwrap(), &["-C", "overflow-checks=off"]);
        assert!(!output.status.success());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.to_lowercase().contains(diagnostic), "{error}");
        assert!(error.contains(&format!("{name}.rs:5:")), "{error}");
        assert!(
            !error.contains("internal compiler error") && !error.contains("panicked"),
            "{error}"
        );
    }
}

#[test]
fn duplicate_entrypoint_names_have_source_diagnostics() {
    let case = support::differential::Case::new(43);
    let source = case.directory.join("duplicate.rs");
    std::fs::write(&source, "#![no_std]\nmod first { use metal_oxide_device::kernel; #[kernel] pub unsafe fn repeated() {} }\nmod second { use metal_oxide_device::kernel; #[kernel] pub unsafe fn repeated() {} }\n").unwrap();
    let (output, _) = support::emit(source.to_str().unwrap(), &[]);
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.contains("duplicate") && error.contains("duplicate.rs:"),
        "{error}"
    );
    assert!(
        !error.contains("internal compiler error") && !error.contains("panicked"),
        "{error}"
    );
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[ignore = "requires a Metal GPU"]
fn seeded_rust_matches_cpu_on_gpu() {
    use metal_oxide::{Argument, Device, LaunchConfig, Module, Pipeline};
    let device = Device::system_default().unwrap();
    for seed in [0, 0xdeadbeef, u32::MAX] {
        let case = support::differential::Case::new(seed);
        let directory = case.compile();
        let module = Module::from_source(
            &device,
            &std::fs::read_to_string(directory.join("kernels.metal")).unwrap(),
        )
        .unwrap();
        let pipeline = Pipeline::new(&device, &module, "seeded").unwrap();
        let input = device.buffer_from_slice(&case.inputs).unwrap();
        let mut output = device.buffer_zeroed::<u32>(case.inputs.len()).unwrap();
        // SAFETY: complete buffers are disjoint; padded threads are guarded and every valid index has one writer.
        unsafe {
            device
                .launch(
                    &pipeline,
                    LaunchConfig::<64>::for_elements(257).unwrap(),
                    &[
                        Argument::read(&input),
                        Argument::write(&mut output),
                        Argument::value(257_u32).unwrap(),
                    ],
                )
                .unwrap();
        }
        assert_eq!(output.as_slice(), case.cpu(), "seed {seed}");
    }
}
