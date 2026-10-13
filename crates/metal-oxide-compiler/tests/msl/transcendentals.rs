use super::support;

#[test]
fn rust_transcendentals_execute_with_distinct_operations_and_arguments() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/transcendentals.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let result = support::execute_msl(
        &directory,
        r#"
        float x[3] = {2.0f, -0.75f, 0.25f};
        float y[3] = {3.0f, 2.0f, -0.5f};
        float out[32];
        for (float &value : out) value = -8192.0f;
        uint n = 3;
        for (uint i = 0; i < 4; ++i)
            transcendentals(x,y,out,n,{i,0,0},{0,0,0},{4,1,1},{1,1,1}, metal_oxide_gate);
        for (uint i = 0; i < 3; ++i) {
            float want[8] = {std::log(x[i]), std::exp(x[i]), std::sin(x[i]),
                std::cos(x[i]), std::pow(x[i], y[i]), std::tanh(x[i]),
                std::atan(x[i]), std::floor(x[i])};
            for (uint j = 0; j < 8; ++j)
                if (out[i*8+j] != want[j] && !(std::isnan(out[i*8+j]) && std::isnan(want[j])))
                    return 1;
        }
        for (uint i = 24; i < 32; ++i) if (out[i] != -8192.0f) return 2;
        std::cout << "ok";
        "#,
    );
    assert_eq!(result, "ok");
}

#[test]
fn sin_cos_emits_one_sincos_call_in_each_math_mode() {
    for mode in ["precise", "fast"] {
        let (output, directory) = support::emit(
            "crates/metal-oxide-compiler/tests/fixtures/sin_cos.rs",
            &["-C", "overflow-checks=off", "--metal-math-mode", mode],
        );
        support::checked(output);
        let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
        assert_eq!(msl.matches("::sincos(").count(), 1, "{msl}");
        assert!(
            msl.contains(&format!("{mode}::sincos(value, cosine)")),
            "{msl}"
        );
        let result = support::execute_msl(
            &directory,
            r#"
            float x[4] = {-0.0f, 0.5f, -3.0f, 100.0f};
            float out[17];
            out[16] = -8192.0f;
            uint n = 4;
            for (uint i = 0; i < 5; ++i)
                sin_cos_pairs(x, out, n, {i,0,0}, {0,0,0}, {5,1,1}, {1,1,1}, metal_oxide_gate);
            for (uint i = 0; i < 16; ++i) {
                float want = i % 2 == 0 ? std::sin(x[i / 4]) : std::cos(x[i / 4]);
                if (as_type<uint>(out[i]) != as_type<uint>(want)) return 1;
            }
            if (out[16] != -8192.0f) return 2;
            std::cout << "ok";
            "#,
        );
        assert_eq!(result, "ok");
    }
}

#[test]
fn box_muller_helper_and_exponential_step_execute_from_rust() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/transcendentals.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let result = support::execute_msl(
        &directory,
        r#"
        float u[3] = {0.25f, 0.5f, 1.0f};
        float v[3] = {0.125f, 0.375f, 0.75f};
        float out[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
        uint n = 3;
        for (uint i = 0; i < 4; ++i)
            box_muller_step(u,v,out,n,{i,0,0},{0,0,0},{4,1,1},{1,1,1}, metal_oxide_gate);
        for (uint i = 0; i < 3; ++i) {
            float z = std::sqrt(-2.0f * std::log(u[i])) * std::cos(6.283185307179586f * v[i]);
            float want = 100.0f * std::exp(0.01f + 0.2f * z);
            if (std::fabs(out[i] - want) > 2e-5f) return 1;
        }
        if (out[3] != -1.0f) return 2;
        std::cout << "ok";
        "#,
    );
    assert_eq!(result, "ok");
}

#[test]
fn rounding_matches_rust_including_halfway_cases_and_signed_zero() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/rounding.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let inputs = [
        -2.5_f32,
        -1.5,
        -0.5,
        -0.4,
        -0.0,
        0.0,
        0.4,
        0.5,
        1.5,
        2.5,
        3.7,
        -3.7,
        8_388_607.5,
        1e30,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let bits = |value: f32| format!("as_type<float>(0x{:08x}u)", value.to_bits());
    let want = inputs
        .iter()
        .flat_map(|x| [x.round(), x.ceil(), x.trunc()])
        .map(bits)
        .collect::<Vec<_>>();
    let n = inputs.len();
    let result = support::execute_msl(
        &directory,
        &format!(
            r#"
            float x[{n}] = {{{inputs}}};
            float want[{m}] = {{{want}}};
            float out[{m}];
            uint n = {n};
            for (uint i = 0; i < n; ++i) rounding(x, out, n, {{i,0,0}}, {{0,0,0}}, {{n,1,1}}, {{1,1,1}}, metal_oxide_gate);
            for (uint i = 0; i < {m}; ++i) {{
                bool same = as_type<uint>(out[i]) == as_type<uint>(want[i]);
                if (!same && !(std::isnan(out[i]) && std::isnan(want[i]))) {{ std::cout << i; return 1; }}
            }}
            std::cout << "ok";
            "#,
            inputs = inputs.map(bits).join(", "),
            m = 3 * n,
            want = want.join(", "),
        ),
    );
    assert_eq!(result, "ok");
}
