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
            transcendentals(x,y,out,n,{i,0,0},{0,0,0},{4,1,1},{1,1,1});
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
            box_muller_step(u,v,out,n,{i,0,0},{0,0,0},{4,1,1},{1,1,1});
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
