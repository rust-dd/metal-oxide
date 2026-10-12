use super::support;

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/integer_bits.rs";
const PATTERNS: [u64; 14] = [
    0,
    1,
    2,
    3,
    0x7f,
    0x80,
    0xff,
    0x0123_4567_89ab_cdef,
    0x5555_5555_5555_5555,
    0x7fff_ffff_ffff_ffff,
    0x8000_0000_0000_0000,
    0xaaaa_aaaa_aaaa_aaaa,
    0xfedc_ba98_7654_3210,
    u64::MAX,
];
const SHIFTS: [u32; 14] = [0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 100, u32::MAX];

/// Runs one `bits_*` kernel from emitted MSL and compares it with Rust's results.
fn run(kernel: &str, ty: &str, inputs: &[String], counts: &[u32], values: &[String]) {
    let (output, directory) = support::emit(FIXTURE, &["-C", "overflow-checks=off"]);
    support::checked(output);
    let n = SHIFTS.len();
    let list = |items: &[String]| items.join(", ");
    let shifts = SHIFTS.map(|s| format!("{s}u")).join(", ");
    let counts = counts
        .iter()
        .map(|c| format!("{c}u"))
        .collect::<Vec<_>>()
        .join(", ");
    let result = support::execute_msl(
        &directory,
        &format!(
            r#"
            {ty} input[{inputs}] = {{{input_list}}};
            uint shifts[{n}] = {{{shifts}}};
            uint counts[{counts_len}] = {{}};
            {ty} values[{values_len}] = {{}};
            uint want_counts[{counts_len}] = {{{counts}}};
            {ty} want_values[{values_len}] = {{{value_list}}};
            uint n = {n};
            for (uint i = 0; i < n; ++i)
                {kernel}(input, shifts, counts, values, n, {{i,0,0}}, {{0,0,0}}, {{n,1,1}}, {{1,1,1}});
            for (uint i = 0; i < {counts_len}; ++i) if (counts[i] != want_counts[i]) {{ std::cout << "count " << i; return 1; }}
            for (uint i = 0; i < {values_len}; ++i) if (values[i] != want_values[i]) {{ std::cout << "value " << i; return 2; }}
            std::cout << "ok";
            "#,
            inputs = inputs.len(),
            input_list = list(inputs),
            counts_len = 3 * n,
            values_len = 5 * n,
            value_list = list(values),
        ),
    );
    assert_eq!(result, "ok", "{kernel}");
}

macro_rules! check {
    ($test:ident, $kernel:literal, $ty:ty, $unsigned:ty, $cpp:literal, $ucpp:literal) => {
        #[test]
        fn $test() {
            let literal = |value: $ty| format!("{}({}({:#x}ull))", $cpp, $ucpp, value as $unsigned);
            let patterns = PATTERNS.map(|p| p as $unsigned as $ty);
            let (mut inputs, mut counts, mut values) = (Vec::new(), Vec::new(), Vec::new());
            for (k, &shift) in SHIFTS.iter().enumerate() {
                let a = patterns[k];
                let b = patterns[(k * 5 + 3) % patterns.len()];
                inputs.extend([literal(a), literal(b)]);
                counts.extend([a.leading_zeros(), a.trailing_zeros(), a.count_ones()]);
                values.extend(
                    [
                        a.reverse_bits(),
                        a.rotate_left(shift),
                        a.rotate_right(shift),
                        a.min(b),
                        a.max(b),
                    ]
                    .map(literal),
                );
            }
            run($kernel, $cpp, &inputs, &counts, &values);
        }
    };
}

check!(
    u8_bit_operations_match_rust,
    "bits_u8",
    u8,
    u8,
    "uchar",
    "uchar"
);
check!(
    u16_bit_operations_match_rust,
    "bits_u16",
    u16,
    u16,
    "ushort",
    "ushort"
);
check!(
    u32_bit_operations_match_rust,
    "bits_u32",
    u32,
    u32,
    "uint",
    "uint"
);
check!(
    u64_bit_operations_match_rust,
    "bits_u64",
    u64,
    u64,
    "ulong",
    "ulong"
);
check!(
    i8_bit_operations_match_rust,
    "bits_i8",
    i8,
    u8,
    "char",
    "uchar"
);
check!(
    i16_bit_operations_match_rust,
    "bits_i16",
    i16,
    u16,
    "short",
    "ushort"
);
check!(
    i32_bit_operations_match_rust,
    "bits_i32",
    i32,
    u32,
    "int",
    "uint"
);
