use super::support;

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/range_loops.rs";
const BOUNDS: [(i32, i32); 9] = [
    (0, 0),
    (3, 1),
    (-3, 3),
    (0, 1),
    (5, 6),
    (-10, -2),
    (0, 100),
    (i32::MAX - 2, i32::MAX),
    (i32::MIN, i32::MIN + 3),
];
const LIMITS: [u8; 8] = [0, 3, 5, 6, 200, 254, 255, 128];

fn list(values: impl IntoIterator<Item = impl ToString>) -> String {
    values
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Runs `main` against the fixture's MSL, compiled without overflow checks like release kernels.
fn execute(main: &str) -> String {
    let (output, directory) = support::emit(FIXTURE, &["-C", "overflow-checks=off"]);
    support::checked(output);
    support::execute_msl(&directory, main)
}

#[test]
fn range_loops_cover_empty_reversed_signed_and_extreme_bounds() {
    let n = BOUNDS.len();
    let sums = BOUNDS.map(|(start, end)| (start..end).fold(0_i32, i32::wrapping_add));
    let counts = BOUNDS.map(|(start, end)| (start..end).count() as u32);
    let bounds = BOUNDS.iter().flat_map(|&(start, end)| [start, end]);
    let result = execute(&format!(
        r#"
        int bounds[{pairs}] = {{{bounds}}};
        int sums[{n}] = {{}};
        uint counts[{n}] = {{}};
        int want_sums[{n}] = {{{sums}}};
        uint want_counts[{n}] = {{{counts}}};
        for (uint i = 0; i < {n}; ++i)
            range_sums(bounds, sums, counts, {{i,0,0}}, {{0,0,0}}, {{{n},1,1}}, {{1,1,1}});
        for (uint i = 0; i < {n}; ++i)
            if (sums[i] != want_sums[i] || counts[i] != want_counts[i]) {{ std::cout << i; return 1; }}
        std::cout << "ok";
        "#,
        pairs = 2 * n,
        bounds = list(bounds.map(|value| format!("int({value}ll)"))),
        sums = list(sums.map(|value| format!("int({value}ll)"))),
        counts = list(counts),
    ));
    assert_eq!(result, "ok");
}

#[test]
fn nested_range_loops_index_a_proved_local_array() {
    let offset = 7_u32;
    let mut table = [0_u32; 16];
    for row in 0..4 {
        for column in row..4 {
            table[row * 4 + column] = (row * 10 + column) as u32 + offset;
        }
    }
    let result = execute(&format!(
        r#"
        uint output[32] = {{}};
        uint offset = {offset};
        uint want[16] = {{{table}}};
        for (uint i = 0; i < 2; ++i)
            nested_ranges(output, offset, {{i,0,0}}, {{0,0,0}}, {{2,1,1}}, {{1,1,1}});
        for (uint i = 0; i < 32; ++i) if (output[i] != want[i % 16]) {{ std::cout << i; return 1; }}
        std::cout << "ok";
        "#,
        table = list(table),
    ));
    assert_eq!(result, "ok");
}

#[test]
fn nested_runtime_ranges_prove_guarded_array_indices() {
    let counts = [0_u32, 1, 2, 5, 16, 17, 100];
    let want = counts.iter().flat_map(|&count| {
        let mut values = [0_u32; 16];
        if (2..=16).contains(&count) {
            for attempt in 0..3 {
                values[0] = attempt;
                for node in 1..count {
                    values[node as usize] = values[(node - 1) as usize] + node * 3;
                }
                if values[(count - 1) as usize] >= 40 {
                    break;
                }
            }
        }
        values
    });
    let result = execute(&format!(
        r#"
        uint counts[{n}] = {{{counts}}};
        uint want[{cells}] = {{{want}}};
        for (uint c = 0; c < {n}; ++c) {{
            uint output[16] = {{}};
            guarded_ranges(counts[c], 3u, output, {{0,0,0}}, {{0,0,0}}, {{1,1,1}}, {{1,1,1}});
            for (uint k = 0; k < 16; ++k) if (output[k] != want[16 * c + k]) {{ std::cout << c << ',' << k; return 1; }}
        }}
        std::cout << "ok";
        "#,
        n = counts.len(),
        cells = 16 * counts.len(),
        counts = list(counts),
        want = list(want),
    ));
    assert_eq!(result, "ok");
}

#[test]
fn byte_ranges_stop_below_the_type_maximum() {
    let n = LIMITS.len();
    let want = LIMITS.iter().flat_map(|&limit| {
        [
            (250_u8..255).map(u32::from).sum::<u32>(),
            (limit..u8::MAX).map(|k| u32::from(k) + 1).sum(),
            (5_u8..limit).map(|k| u32::from(k) * 3).sum(),
        ]
    });
    let result = execute(&format!(
        r#"
        uchar limits[{n}] = {{{limits}}};
        uint output[{values}] = {{}};
        uint want[{values}] = {{{want}}};
        for (uint i = 0; i < {n}; ++i)
            byte_ranges(limits, output, {{i,0,0}}, {{0,0,0}}, {{{n},1,1}}, {{1,1,1}});
        for (uint i = 0; i < {values}; ++i) if (output[i] != want[i]) {{ std::cout << i; return 1; }}
        std::cout << "ok";
        "#,
        values = 3 * n,
        limits = list(LIMITS),
        want = list(want),
    ));
    assert_eq!(result, "ok");
}

#[test]
fn explicit_next_calls_advance_a_local_range() {
    let n = 40_u32;
    let want = (0..n).map(|i| (i..n).fold(1_u32, |product, k| product.wrapping_mul(k + 2)));
    let result = execute(&format!(
        r#"
        uint output[{n}] = {{}};
        uint n = {n};
        uint want[{n}] = {{{want}}};
        for (uint i = 0; i < n; ++i)
            manual_next(output, n, {{i,0,0}}, {{0,0,0}}, {{n,1,1}}, {{1,1,1}});
        for (uint i = 0; i < n; ++i) if (output[i] != want[i]) {{ std::cout << i; return 1; }}
        std::cout << "ok";
        "#,
        want = list(want),
    ));
    assert_eq!(result, "ok");
}
