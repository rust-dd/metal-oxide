#![cfg(feature = "rustc-private")]

#[path = "core_methods/model.rs"]
mod model;
mod support;

use std::{fmt::Write, path::PathBuf};

fn emitted() -> PathBuf {
    let (output, directory) = support::emit(model::FIXTURE, &["-C", "overflow-checks=off"]);
    support::checked(output);
    directory
}

const LAUNCH: &str = "{}, {}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate";

#[test]
fn arbitrary_diverging_helpers_are_not_treated_as_unreachable_hints() {
    let (output, _) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/rejections/diverging_call.rs",
        &[],
    );
    support::rejected(
        output,
        "calls to functions that never return are supported only for panics",
    );
}

fn compare(directory: &std::path::Path, source: &str, expected: &[u64]) {
    let output = support::execute_msl(directory, source);
    let actual = output
        .split(',')
        .filter(|word| !word.is_empty())
        .map(|word| word.parse::<u64>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn integer_powers_comparisons_division_and_byte_swaps_match_rust() {
    let directory = emitted();
    let mut source = String::new();
    let mut expected = Vec::new();
    for &(a, b, exponent) in model::POWERS {
        write!(source, "{{ uint out[18] = {{}}; powers(out, {a}u, as_type<int>({}u), {exponent}u, {LAUNCH}); for (auto value : out) std::cout << value << ','; }}", b as u32).unwrap();
        expected.extend(model::powers(a, b, exponent).map(u64::from));
    }
    for &(a, exponent) in model::WIDE_POWERS {
        write!(source, "{{ ulong out[5] = {{}}; wide_powers(out, {a}ul, {exponent}u, {LAUNCH}); for (auto value : out) std::cout << value << ','; }}").unwrap();
        expected.extend(model::wide_powers(a, exponent));
    }
    for &(a, b, c, d) in model::INTEGERS {
        write!(source, "{{ int out[7] = {{}}; comparisons(out, {a}u, {b}u, as_type<int>({}u), as_type<int>({}u), {LAUNCH}); for (auto value : out) std::cout << uint(value) << ','; }}", c as u32, d as u32).unwrap();
        expected.extend(model::comparisons(a, b, c, d).map(|n| n as u32 as u64));
        write!(source, "{{ uint out[8] = {{}}; checked_division(out, {a}u, {b}u, as_type<int>({}u), as_type<int>({}u), {LAUNCH}); for (auto value : out) std::cout << value << ','; }}", c as u32, d as u32).unwrap();
        expected.extend(model::checked_division(a, b, c, d).map(u64::from));
    }
    for &value in model::BITS {
        write!(source, "{{ ulong out[11] = {{}}; bit_methods(out, {value}ul, {LAUNCH}); for (auto value : out) std::cout << value << ','; }}").unwrap();
        expected.extend(model::bit_methods(value));
    }
    for &a in model::BITS {
        for &b in model::BITS {
            write!(source, "{{ int out[10] = {{}}; primitive_ordering(out, {a}ul, {b}ul, {LAUNCH}); for (auto value : out) std::cout << uint(value) << ','; }}").unwrap();
            expected.extend(model::primitive_ordering(a, b).map(|n| n as u32 as u64));
        }
    }
    source.push_str("std::cout << metal_oxide_failure_word.load();");
    expected.push(0);
    compare(&directory, &source, &expected);
}

#[test]
fn float_signs_and_ordering_match_rust_bits() {
    let directory = emitted();
    let mut source = String::new();
    let mut expected = Vec::new();
    for &a in model::FLOATS {
        for &b in model::FLOATS {
            write!(source, "{{ uint out[4] = {{}}; float_methods(out, as_type<float>({a}u), as_type<float>({b}u), {LAUNCH}); for (auto value : out) std::cout << value << ','; }}").unwrap();
            expected.extend(model::float_methods(a, b).map(u64::from));
        }
    }
    for &a in model::SUBNORMALS {
        for &b in model::FLOATS {
            write!(source, "{{ uint out[3] = {{}}; float_bit_methods(out, as_type<float>({a}u), as_type<float>({b}u), {LAUNCH}); for (auto value : out) std::cout << value << ','; }}").unwrap();
            expected.extend(
                model::float_methods(a, b)[..3]
                    .iter()
                    .copied()
                    .map(u64::from),
            );
        }
    }
    compare(&directory, &source, &expected);
}

#[test]
fn unchecked_unwraps_keep_no_checks_and_match_rust() {
    let directory = emitted();
    let checks = serde_json::from_slice::<Vec<serde_json::Value>>(
        &std::fs::read(directory.join("checks.json")).unwrap(),
    )
    .unwrap();
    // The kernel spans lines 165-177 of the fixture; its `None` and `Err` arms never run.
    assert!(
        checks
            .iter()
            .all(|check| !(165..=177).contains(&check["location"]["line"].as_u64().unwrap())),
        "{checks:?}"
    );
    let mut source = String::new();
    let mut expected = Vec::new();
    for (a, b) in [(0_u32, 7_u32), (5, 100), (255, u32::MAX), (u32::MAX - 1, 3)] {
        write!(source, "{{ uint out[3] = {{}}; unchecked_unwraps(out, {a}u, {b}u, {LAUNCH}); for (auto value : out) std::cout << value << ','; }}").unwrap();
        expected.extend([a + 1, b / (a | 1), a & 0xff].map(u64::from));
    }
    compare(&directory, &source, &expected);
}

#[test]
fn clamp_results_and_invalid_bounds_preserve_rust_behavior() {
    let directory = emitted();
    let checks = serde_json::from_slice::<Vec<serde_json::Value>>(
        &std::fs::read(directory.join("checks.json")).unwrap(),
    )
    .unwrap();
    for message in ["min > max", "min > max, or either was NaN"] {
        assert!(
            checks
                .iter()
                .any(|check| check["kind"] == serde_json::json!({"panic": message})),
            "{checks:?}"
        );
    }
    let mut source = String::new();
    let mut expected = Vec::new();
    for &(value, low, high) in model::FLOAT_CLAMPS {
        write!(source, "{{ metal_oxide_failure_word.store(0); float out[1] = {{123.0f}}; float_clamp(out, as_type<float>({}u), as_type<float>({}u), as_type<float>({}u), {LAUNCH}, metal_oxide_failure); std::cout << as_type<uint>(out[0]) << ',' << (metal_oxide_failure_word.load() >= 2) << ','; }}", value.to_bits(), low.to_bits(), high.to_bits()).unwrap();
        let valid = low <= high;
        expected.extend([
            if valid { value.clamp(low, high) } else { 123.0 }.to_bits() as u64,
            u64::from(!valid),
        ]);
    }
    for &(value, low, high) in model::UNSIGNED_CLAMPS {
        write!(source, "{{ metal_oxide_failure_word.store(0); uint out[1] = {{123u}}; unsigned_clamp(out, {value}u, {low}u, {high}u, {LAUNCH}, metal_oxide_failure); std::cout << out[0] << ',' << (metal_oxide_failure_word.load() >= 2) << ','; }}").unwrap();
        expected.extend([
            if low <= high {
                value.clamp(low, high)
            } else {
                123
            } as u64,
            u64::from(low > high),
        ]);
    }
    for &(value, low, high) in model::SIGNED_CLAMPS {
        write!(source, "{{ metal_oxide_failure_word.store(0); int out[1] = {{123}}; signed_clamp(out, as_type<int>({}u), as_type<int>({}u), as_type<int>({}u), {LAUNCH}, metal_oxide_failure); std::cout << uint(out[0]) << ',' << (metal_oxide_failure_word.load() >= 2) << ','; }}", value as u32, low as u32, high as u32).unwrap();
        expected.extend([
            if low <= high {
                value.clamp(low, high)
            } else {
                123
            } as u32 as u64,
            u64::from(low > high),
        ]);
    }
    compare(&directory, &source, &expected);
}

#[test]
fn ordinary_power_preserves_enabled_overflow_checks() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/core_pow_checked.rs",
        &["-C", "overflow-checks=on"],
    );
    support::checked(output);
    let source = format!(
        "uint out[1] = {{123u}}; power(out, 2u, 30u, {LAUNCH}, metal_oxide_failure); std::cout << out[0] << ',' << metal_oxide_failure_word.load() << ','; out[0] = 123u; power(out, 2u, 32u, {LAUNCH}, metal_oxide_failure); std::cout << out[0] << ',' << (metal_oxide_failure_word.load() >= 2);"
    );
    compare(&directory, &source, &[1 << 30, 0, 123, 1]);
}

#[test]
fn branch_hints_preserve_conditions_and_static_knowledge_is_conservative() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/core_hints.rs",
        &[],
    );
    support::checked(output);
    let checks = std::fs::read_to_string(directory.join("checks.json")).unwrap();
    assert_eq!(checks.trim(), "[]");
    let source = format!(
        "uint out[3] = {{}}; for (uint divisor : {{0u, 1u, 3u, 4294967295u}}) {{ hints(out, 7u, divisor, {LAUNCH}); for (auto value : out) std::cout << value << ','; }}"
    );
    compare(&directory, &source, &[0, 0, 0, 0, 7, 0, 0, 2, 1, 0, 0, 7]);
}
