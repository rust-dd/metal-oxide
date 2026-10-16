use nostd_numeric_kernels::{float_values, integer_values, libm_values};

use super::{assert_float, kernels, support};

#[test]
fn imported_library_workloads_execute_in_emitted_msl() {
    let directory = kernels();
    let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    let floats = super::float_inputs().first_chunk::<8>().copied().unwrap();
    let initializers = floats
        .iter()
        .map(|value: &f32| format!("as_type<float>({}u)", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",");
    let mut program = invocation(&msl, "libm_f32", "float", &initializers, "", 8);
    program.push_str(&invocation(
        &msl,
        "traits_f32",
        "float",
        &initializers,
        ", -3",
        4,
    ));
    let mut integers = Vec::new();
    macro_rules! add {
        ($ty:ident, $metal:literal) => {
            let values = [
                <$ty>::MIN,
                0,
                1,
                2,
                -3_i32 as $ty,
                -1_i32 as $ty,
                <$ty>::MAX / 2,
                <$ty>::MAX,
            ];
            let initializers = values
                .iter()
                .map(|value| format!("{}({}ul)", $metal, *value as u64))
                .collect::<Vec<_>>()
                .join(",");
            for exponent in [0, 2, 3, 31, 64] {
                program.push_str(&invocation(
                    &msl,
                    concat!("traits_", stringify!($ty)),
                    $metal,
                    &initializers,
                    &format!(", {exponent}u"),
                    4,
                ));
                integers.extend(
                    values.into_iter().flat_map(|value| {
                        integer_values(value, exponent).map(|result| result as u64)
                    }),
                );
            }
        };
    }
    add!(i8, "char");
    add!(i16, "short");
    add!(i32, "int");
    add!(u8, "uchar");
    add!(u16, "ushort");
    add!(u32, "uint");
    add!(u64, "ulong");
    let output = support::execute_msl(directory, &program);
    let mut actual = output
        .split_whitespace()
        .map(|word| word.parse::<u64>().unwrap());
    for expected in floats
        .into_iter()
        .flat_map(libm_values)
        .chain(floats.into_iter().flat_map(|value| float_values(value, -3)))
    {
        assert_float(
            f32::from_bits(actual.next().unwrap() as u32),
            expected,
            "emitted library f32 workload",
        );
    }
    assert_eq!(actual.collect::<Vec<_>>(), integers);
}

fn invocation(
    msl: &str,
    entry: &str,
    scalar: &str,
    initializers: &str,
    exponent: &str,
    lanes: usize,
) -> String {
    let signature = msl
        .lines()
        .find(|line| line.starts_with(&format!("kernel void {entry}(")))
        .unwrap();
    let output = signature
        .split("device ")
        .nth(2)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap();
    let cast = if scalar == "float" {
        "as_type<uint>"
    } else {
        "ulong"
    };
    format!(
        "{{ {scalar} input[8] = {{{initializers}}}; {output} output[8] = {{}}; \
         ulong lengths[3] = {{8, 8, 0}}; \
         for(uint i=0;i<8;i++) {entry}(input, output{exponent}, \
         {{i,0,0}}, {{}}, {{8,1,1}}, {{1,1,1}}, {{i,0,0}}, metal_oxide_gate, lengths); \
         for(uint i=0;i<8;i++) for(uint j=0;j<{lanes};j++) \
         std::cout << {cast}(output[i].elements[j]) << ' '; }}\n"
    )
}
