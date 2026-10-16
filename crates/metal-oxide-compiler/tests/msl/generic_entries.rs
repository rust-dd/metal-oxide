use super::support;
use metal_oxide_artifact::{Abi, Access, Layout, ParameterType, Scalar};

#[test]
fn generic_entries_keep_concrete_types_constants_names_and_contracts() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/generic_entries.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let abi =
        Abi::from_json(&std::fs::read_to_string(directory.join("abi.json")).unwrap()).unwrap();
    assert_eq!(abi.kernels.len(), 4);
    for (name, scalar) in [
        ("scale_f32", Scalar::F32),
        ("scale_i32", Scalar::I32),
        ("scale_u32", Scalar::U32),
        ("alias_f32", Scalar::F32),
    ] {
        let kernel = abi
            .kernels
            .iter()
            .find(|kernel| kernel.name == name)
            .unwrap();
        assert!(kernel.safe);
        assert!(kernel.one_dimensional);
        assert!(kernel.lengths);
        assert!(!kernel.fallible);
        assert_eq!(kernel.required_block, Some([32, 1, 1]));
        assert_eq!(kernel.requires.len(), 1);
        for (parameter, name, access) in [
            (&kernel.parameters[0], "input", Access::Read),
            (&kernel.parameters[1], "output", Access::Write),
        ] {
            assert_eq!(parameter.name, name);
            assert_eq!(
                parameter.ty,
                ParameterType::Buffer {
                    element: Layout::scalar(scalar),
                    stride: 4,
                    access,
                }
            );
        }
    }
    assert_eq!(
        support::execute_msl(
            &directory,
            r#"
            float f[3] = {1.5f, -2.0f, 0.0f}, fo[3] = {}, alias[3] = {};
            int i[3] = {-3, 4, 0}, io[3] = {};
            uint u[3] = {7, 0, 4294967295u}, uo[3] = {};
            ulong lengths[2] = {3, 3};
            for (uint lane = 0; lane < 32; ++lane) {
                scale_f32(f, fo, {lane,0,0}, {}, {32,1,1}, {1,1,1}, {lane,0,0}, metal_oxide_gate, lengths);
                alias_f32(f, alias, {lane,0,0}, {}, {32,1,1}, {1,1,1}, {lane,0,0}, metal_oxide_gate, lengths);
                scale_i32(i, io, {lane,0,0}, {}, {32,1,1}, {1,1,1}, {lane,0,0}, metal_oxide_gate, lengths);
                scale_u32(u, uo, {lane,0,0}, {}, {32,1,1}, {1,1,1}, {lane,0,0}, metal_oxide_gate, lengths);
            }
            for (auto v : fo) std::cout << v << ',';
            for (auto v : alias) std::cout << v << ',';
            for (auto v : io) std::cout << v << ',';
            for (auto v : uo) std::cout << v << ',';
            "#,
        ),
        "6,-8,0,6,-8,0,-24,32,0,14,0,4294967294,"
    );
}
