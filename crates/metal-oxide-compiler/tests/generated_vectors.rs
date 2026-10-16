#![cfg(feature = "rustc-private")]

mod generated_support;
mod support;
mod vector_generated;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "vector_generated/gpu.rs"]
mod gpu;

use std::collections::BTreeSet;
use vector_generated::{Program, model, programs};

#[test]
fn generated_vector_programs_cover_operations_and_vary_by_seed() {
    for scalar in model::SCALARS {
        for lanes in 2..=4 {
            let mut sources = BTreeSet::new();
            for seed in model::SEEDS {
                let program = Program::generate(seed, scalar, lanes);
                assert_eq!(
                    program.source(),
                    Program::generate(seed, scalar, lanes).source()
                );
                let coverage = program.coverage();
                let selectors = program.inputs.iter().map(|words| {
                    if scalar == "f32" {
                        words[0] >> 17
                    } else {
                        words[0]
                    }
                });
                assert_eq!(
                    selectors
                        .clone()
                        .map(|word| word as usize % lanes)
                        .collect::<BTreeSet<_>>()
                        .len(),
                    lanes,
                    "{} lacks dynamic lanes",
                    program.name("coverage")
                );
                assert_eq!(
                    selectors
                        .map(|word| word & 1)
                        .collect::<BTreeSet<_>>()
                        .len(),
                    2,
                    "{} lacks both mask values",
                    program.name("coverage")
                );
                for op in required_operations(scalar) {
                    assert!(
                        coverage.contains(&op),
                        "{} lacks {op:?}",
                        program.name("coverage")
                    );
                }
                sources.insert(program.source());
            }
            assert_eq!(sources.len(), model::SEEDS.len());
        }
    }
}

fn required_operations(scalar: &str) -> Vec<String> {
    let mut required = [
        "Array",
        "Splat",
        "ArrayReference",
        "Insert",
        "Swizzle(false)",
        "Swizzle(true)",
        "Reverse",
        "Rotate(false)",
        "Rotate(true)",
    ]
    .map(str::to_owned)
    .to_vec();
    let mut add = |family, names: &[&str]| {
        required.extend(names.iter().map(|name| format!("{family}({name:?})")));
    };
    add("Binary", &["+", "-", "*", "/", "%"]);
    add("Method", &["simd_min", "simd_max", "simd_clamp"]);
    add(
        "Compare",
        &[
            "simd_eq", "simd_ne", "simd_lt", "simd_le", "simd_gt", "simd_ge",
        ],
    );
    add(
        "Mask",
        &[
            "array", "splat", "bitmask", "and", "or", "xor", "not", "any", "all",
        ],
    );
    add(
        "Reduction",
        &["reduce_sum", "reduce_product", "reduce_min", "reduce_max"],
    );
    add("Cast", &model::SCALARS);
    if scalar != "f32" {
        add("Binary", &["&", "|", "^", "<<", ">>"]);
        add(
            "Method",
            &[
                "not",
                "saturating_add",
                "saturating_sub",
                "reverse_bits",
                "count_ones",
                "count_zeros",
                "leading_zeros",
                "trailing_zeros",
                "leading_ones",
                "trailing_ones",
            ],
        );
        add("Reduction", &["reduce_and", "reduce_or", "reduce_xor"]);
    }
    if !scalar.starts_with('u') {
        add("Method", &["neg", "abs"]);
    }
    if scalar.starts_with('i') {
        add("Method", &["saturating_abs", "saturating_neg"]);
    }
    required
}

#[test]
fn generated_vector_sources_execute_on_the_cpu() {
    for program in programs() {
        let name = program.name("compile");
        let expected = program.cpu(&name);
        assert!(program.mismatch(&expected[0], &expected[0]).is_none());
        let mut changed = expected[0].clone();
        changed[0] ^= 0x0001_0000;
        assert!(program.mismatch(&expected[0], &changed).is_some());
    }
}

#[test]
fn generated_vector_sources_compile_for_the_device() {
    for program in programs() {
        let name = program.name("compile");
        let directory = generated_support::compile(&name, &program.source());
        let abi = metal_oxide_artifact::Abi::from_json(
            &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
        )
        .unwrap();
        assert!(abi.required_features.is_empty(), "{name}");
        let source = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
        for kernel in program.kernels() {
            assert!(source.contains(&format!("kernel void {kernel}(")), "{name}");
        }
    }
}
