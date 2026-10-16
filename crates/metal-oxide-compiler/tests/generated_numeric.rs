#![cfg(feature = "rustc-private")]

mod generated_support;
mod numeric_generated;
mod support;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "numeric_generated/gpu.rs"]
mod gpu;

use numeric_generated::{SEEDS, atomics::FloatProgram, compile, half, wide};
use std::collections::BTreeSet;

#[test]
fn generated_numeric_programs_are_deterministic_and_cover_each_operation() {
    let mut half_sources = BTreeSet::new();
    let mut float_sources = BTreeSet::new();
    let mut wide_sources = BTreeSet::new();
    for seed in SEEDS {
        let half = half::Program::generate(seed);
        let floats = FloatProgram::generate(seed, false);
        let wide = wide::Program::generate(seed);
        assert_eq!(half.source(), half::Program::generate(seed).source());
        assert_eq!(
            floats.source(),
            FloatProgram::generate(seed, false).source()
        );
        assert_eq!(wide.source(), wide::Program::generate(seed).source());
        half_sources.insert(half.source());
        float_sources.insert(floats.source());
        wide_sources.insert(wide.source());
        for ty in ["F16", "BF16", "f16"] {
            let coverage = half.coverage();
            for op in [
                "FromBits",
                "FromFloat",
                "CopyBits",
                "Convert",
                "Negate",
                "PartialCompare",
            ] {
                assert!(
                    coverage.contains(&format!("{ty}:{op}")),
                    "seed {seed:#x} lacks {ty}:{op}"
                );
            }
            for op in ["+", "-", "*", "/", "%"] {
                for assign in [false, true] {
                    assert!(coverage.contains(&format!("{ty}:Binary({op:?}, {assign})")));
                }
            }
            for op in ["==", "!=", "<", "<=", ">", ">="] {
                assert!(coverage.contains(&format!("{ty}:Compare({op:?})")));
            }
        }
        for op in [
            "load",
            "store",
            "exchange",
            "fetch_add",
            "fetch_sub",
            "compare_exchange_weak",
        ] {
            assert!(floats.coverage().contains(&format!("{op}:Relaxed:Relaxed")));
        }
        assert!(wide.source().contains("let () = cell.min("));
        assert!(wide.source().contains("let () = cell.max("));
    }
    assert_eq!(half_sources.len(), SEEDS.len());
    assert_eq!(float_sources.len(), SEEDS.len());
    assert_eq!(wide_sources.len(), SEEDS.len());
}

#[test]
fn generated_numeric_sources_compile_with_reproducible_artifacts() {
    for seed in SEEDS {
        for (kind, source, features) in [
            (
                "half",
                half::Program::generate(seed).source(),
                vec!["bfloat16"],
            ),
            (
                "float",
                FloatProgram::generate(seed, false).source(),
                vec!["float32_atomics"],
            ),
            (
                "wide",
                wide::Program::generate(seed).source(),
                vec!["uint64_atomics"],
            ),
        ] {
            let name = format!("compile-{kind}-{seed:016x}");
            let directory = compile(&name, &source);
            let abi = metal_oxide_artifact::Abi::from_json(
                &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(abi.required_features, features, "{name}");
            assert_eq!(abi.msl_version(), "3.2", "{name}");
            if seed == SEEDS[0] {
                let original = metal_oxide_artifact::COMPILER_OUTPUTS
                    .map(|file| std::fs::read(directory.join(file.name())).unwrap());
                let rebuilt = compile(&name, &source);
                for (file, expected) in metal_oxide_artifact::COMPILER_OUTPUTS
                    .into_iter()
                    .zip(original)
                {
                    assert_eq!(
                        std::fs::read(rebuilt.join(file.name())).unwrap(),
                        expected,
                        "{name}: {}",
                        file.name()
                    );
                }
            }
        }
    }
}

#[test]
fn generated_float_atomic_orderings_require_msl_41() {
    let mut sources = BTreeSet::new();
    for seed in SEEDS {
        let program = FloatProgram::generate(seed, true);
        assert_eq!(
            program.source(),
            FloatProgram::generate(seed, true).source()
        );
        sources.insert(program.source());
        let coverage = program.coverage();
        for op in ["exchange", "fetch_add", "fetch_sub"] {
            for order in ["Relaxed", "Acquire", "Release", "AcqRel"] {
                assert!(coverage.contains(&format!("{op}:{order}:Relaxed")));
            }
        }
        for (success, failure) in [
            ("Relaxed", "Relaxed"),
            ("Acquire", "Relaxed"),
            ("Acquire", "Acquire"),
            ("Release", "Relaxed"),
            ("AcqRel", "Relaxed"),
            ("AcqRel", "Acquire"),
        ] {
            assert!(coverage.contains(&format!("compare_exchange_weak:{success}:{failure}")));
        }
        for ordering in [
            "load:Relaxed:Relaxed",
            "load:Acquire:Relaxed",
            "store:Relaxed:Relaxed",
            "store:Release:Relaxed",
        ] {
            assert!(coverage.contains(ordering));
        }
        let directory = compile(&format!("orders-{seed:016x}"), &program.source());
        let abi = metal_oxide_artifact::Abi::from_json(
            &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            abi.required_features,
            ["float32_atomics", "atomic_acquire_release"]
        );
        assert_eq!(
            abi.target_flags(),
            ["-std=metal4.1", "-mmacosx-version-min=27.0"]
        );
        program.verify_orders(&std::fs::read_to_string(directory.join("kernels.metal")).unwrap());
    }
    assert_eq!(sources.len(), SEEDS.len());
}
