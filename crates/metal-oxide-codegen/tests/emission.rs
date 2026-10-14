mod support;
use metal_oxide_ir::*;
use support::{block, local, module, uint};

#[test]
fn diamond_executes_both_branches_and_shared_return() {
    let module = module(
        1,
        vec![
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::Bool),
        ],
        vec![
            block(
                vec![(2, Expression::Binary(BinaryOp::Lt, local(1), uint(10)))],
                Terminator::Branch {
                    condition: local(2),
                    then_block: 1,
                    else_block: 2,
                },
            ),
            block(
                vec![(0, Expression::Binary(BinaryOp::Add, local(1), uint(3)))],
                Terminator::Goto(3),
            ),
            block(
                vec![(0, Expression::Binary(BinaryOp::Sub, local(1), uint(2)))],
                Terminator::Goto(3),
            ),
            block(vec![], Terminator::Return),
        ],
    );
    assert_eq!(
        support::execute(
            &module,
            "std::cout << metal_oxide_fn_0(9u, ctx) << ',' << metal_oxide_fn_0(12u, ctx);"
        ),
        "12,10"
    );
}

#[test]
fn loop_preserves_carried_values_and_zero_iterations() {
    let module = module(
        1,
        vec![Type::Scalar(Scalar::U32); 4]
            .into_iter()
            .chain([Type::Scalar(Scalar::Bool)])
            .collect(),
        vec![
            block(
                vec![(2, Expression::Use(uint(0))), (3, Expression::Use(uint(0)))],
                Terminator::Goto(1),
            ),
            block(
                vec![(4, Expression::Binary(BinaryOp::Lt, local(2), local(1)))],
                Terminator::Branch {
                    condition: local(4),
                    then_block: 2,
                    else_block: 3,
                },
            ),
            block(
                vec![
                    (3, Expression::Binary(BinaryOp::Add, local(3), local(2))),
                    (2, Expression::Binary(BinaryOp::Add, local(2), uint(1))),
                ],
                Terminator::Goto(1),
            ),
            block(vec![(0, Expression::Use(local(3)))], Terminator::Return),
        ],
    );
    assert_eq!(
        support::execute(
            &module,
            "std::cout << metal_oxide_fn_0(0u, ctx) << ',' << metal_oxide_fn_0(1u, ctx) << ',' << metal_oxide_fn_0(4u, ctx);"
        ),
        "0,0,6"
    );
}

#[test]
fn signed_integer_addition_wraps_without_cpp_signed_overflow() {
    let module = module(
        2,
        vec![Type::Scalar(Scalar::I32); 3],
        vec![block(
            vec![(0, Expression::Binary(BinaryOp::Add, local(1), local(2)))],
            Terminator::Return,
        )],
    );
    assert_eq!(
        support::execute(
            &module,
            "std::cout << metal_oxide_fn_0(2147483647, 1, ctx) << ',' << metal_oxide_fn_0(-2147483647-1, -1, ctx);"
        ),
        "-2147483648,2147483647"
    );
}

#[test]
fn unproved_assertion_records_its_failure_and_returns() {
    let module = assertion(true);
    let codegen = metal_oxide_codegen::Codegen::new(&module).unwrap();
    assert_eq!(codegen.checks().len(), 1);
    assert_eq!(codegen.checks()[0].location.line, support::source().line);
    for (gate, condition, expected) in
        [(0, "true", "0,0"), (0, "false", "2,1"), (1, "false", "0,1")]
    {
        assert_eq!(
            support::execute(
                &module,
                &format!(
                    "gate = {gate}; metal_oxide_fn_0({condition}, ctx); std::cout << atomic_load_explicit(&failure, memory_order_relaxed) << ',' << failed;"
                )
            ),
            expected
        );
    }
}

#[test]
fn explicitly_disabled_optional_assertion_follows_success_edge() {
    assert_eq!(
        support::execute(
            &assertion(false),
            "metal_oxide_fn_0(false, ctx); std::cout << 7;"
        ),
        "7"
    );
}

#[test]
fn values_read_only_by_erased_assertions_are_not_emitted() {
    let module = module(
        1,
        vec![
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::Bool),
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::U32),
        ],
        vec![
            block(
                vec![
                    (2, Expression::Binary(BinaryOp::Lt, local(1), uint(100))),
                    (3, Expression::Binary(BinaryOp::Add, local(1), uint(7))),
                    (4, Expression::Binary(BinaryOp::Mul, local(3), uint(2))),
                    (5, Expression::Binary(BinaryOp::Add, local(1), uint(1))),
                ],
                Terminator::Assert {
                    condition: local(2),
                    expected: true,
                    enabled: false,
                    target: 1,
                    check: Check::Overflow(OverflowOp::Add),
                },
            ),
            block(vec![(0, Expression::Use(local(5)))], Terminator::Return),
        ],
    );
    let msl = metal_oxide_codegen::Codegen::new(&module)
        .and_then(|codegen| codegen.emit())
        .unwrap();
    for local in ["v2", "v3", "v4"] {
        assert!(
            !msl.contains(&format!(" {local};")) && !msl.contains(&format!("{local} = ")),
            "{local}: {msl}"
        );
    }
    assert!(msl.contains("v5 = "), "{msl}");
    assert_eq!(
        support::execute(&module, "std::cout << metal_oxide_fn_0(41u, ctx);"),
        "42"
    );
}

fn assertion(enabled: bool) -> Module {
    module(
        1,
        vec![Type::Unit, Type::Scalar(Scalar::Bool)],
        vec![
            block(
                vec![],
                Terminator::Assert {
                    condition: local(1),
                    expected: true,
                    enabled,
                    target: 1,
                    check: Check::Overflow(OverflowOp::Add),
                },
            ),
            block(vec![], Terminator::Return),
        ],
    )
}

#[test]
fn irreducible_control_flow_is_rejected() {
    let module = module(
        1,
        vec![Type::Unit, Type::Scalar(Scalar::Bool)],
        vec![
            block(
                vec![],
                Terminator::Branch {
                    condition: local(1),
                    then_block: 1,
                    else_block: 2,
                },
            ),
            block(
                vec![],
                Terminator::Branch {
                    condition: local(1),
                    then_block: 2,
                    else_block: 3,
                },
            ),
            block(
                vec![],
                Terminator::Branch {
                    condition: local(1),
                    then_block: 1,
                    else_block: 3,
                },
            ),
            block(vec![], Terminator::Return),
        ],
    );
    assert!(
        metal_oxide_codegen::Codegen::new(&module)
            .and_then(|codegen| codegen.emit())
            .unwrap_err()
            .message
            .contains("control flow")
    );
}

#[test]
fn loop_with_multiple_exits_preserves_each_result() {
    let module = module(
        1,
        vec![Type::Scalar(Scalar::U32), Type::Scalar(Scalar::Bool)],
        vec![
            block(vec![], Terminator::Goto(1)),
            block(
                vec![],
                Terminator::Branch {
                    condition: local(1),
                    then_block: 2,
                    else_block: 4,
                },
            ),
            block(
                vec![],
                Terminator::Branch {
                    condition: local(1),
                    then_block: 3,
                    else_block: 1,
                },
            ),
            block(vec![(0, Expression::Use(uint(11)))], Terminator::Return),
            block(vec![(0, Expression::Use(uint(22)))], Terminator::Return),
        ],
    );
    assert_eq!(
        support::execute(
            &module,
            "std::cout << metal_oxide_fn_0(true, ctx) << ',' << metal_oxide_fn_0(false, ctx);"
        ),
        "11,22",
    );
}

#[test]
fn uniform_switch_can_share_cooperative_cases_with_its_default() {
    let mut module = module(
        1,
        vec![Type::Unit, Type::Scalar(Scalar::U32), Type::Unit],
        vec![
            block(
                vec![],
                Terminator::Switch {
                    discriminant: local(1),
                    cases: vec![
                        (Constant::U32(0), 1),
                        (Constant::U32(1), 2),
                        (Constant::U32(2), 1),
                    ],
                    otherwise: 1,
                },
            ),
            block(
                vec![(2, Expression::ThreadgroupBarrier)],
                Terminator::Return,
            ),
            block(vec![], Terminator::Return),
        ],
    );
    module.functions[0].kernel = true;
    let output = metal_oxide_codegen::Codegen::new(&module)
        .and_then(|codegen| codegen.emit())
        .unwrap();
    assert_eq!(output.matches("threadgroup_barrier(").count(), 1);
}

#[test]
fn float_to_integer_cast_saturates_before_conversion() {
    let module = module(
        1,
        vec![Type::Scalar(Scalar::I32), Type::Scalar(Scalar::F32)],
        vec![block(
            vec![(0, Expression::Cast(local(1), Scalar::I32))],
            Terminator::Return,
        )],
    );
    assert_eq!(
        support::execute(
            &module,
            "for (float value : {as_type<float>(0x7f800000u), as_type<float>(0xff800000u), as_type<float>(0x7fc00000u), 123.75f, -123.75f}) std::cout << metal_oxide_fn_0(value, ctx) << ',';"
        ),
        "2147483647,-2147483648,0,123,-123,"
    );
}

#[test]
fn unit_moves_do_not_reference_undeclared_locals() {
    let helper = module(0, vec![Type::Unit], vec![block(vec![], Terminator::Return)])
        .functions
        .remove(0);
    let mut module = module(
        0,
        vec![Type::Unit; 2],
        vec![block(
            vec![
                (
                    1,
                    Expression::Call {
                        function: 1,
                        arguments: vec![],
                    },
                ),
                (0, Expression::Use(local(1))),
            ],
            Terminator::Return,
        )],
    );
    module.functions.push(helper);
    assert_eq!(
        support::execute(&module, "metal_oxide_fn_0(ctx); std::cout << 5;"),
        "5"
    );
}

#[test]
fn reserved_msl_kernel_names_have_source_diagnostics() {
    for name in [
        "if",
        "while",
        "return",
        "delete",
        "and",
        "_Reserved",
        "has__reserved",
        "uint3",
        "metal",
    ] {
        let mut module = module(0, vec![Type::Unit], vec![block(vec![], Terminator::Return)]);
        module.functions[0].kernel = true;
        module.functions[0].name = name.into();
        let error = metal_oxide_codegen::Codegen::new(&module)
            .and_then(|codegen| codegen.emit())
            .unwrap_err();
        assert_eq!(error.source, support::source());
        assert!(error.message.contains("identifier"));
    }
}

#[test]
fn kernel_bindings_fit_metal_slots_without_limiting_helpers() {
    for (parameters, kernel, valid) in [(28, true, true), (29, true, false), (32, false, true)] {
        let mut module = module(
            parameters,
            std::iter::once(Type::Unit)
                .chain(std::iter::repeat_n(Type::Scalar(Scalar::U32), parameters))
                .collect(),
            vec![block(vec![], Terminator::Return)],
        );
        module.functions[0].kernel = kernel;
        match metal_oxide_codegen::Codegen::new(&module).and_then(|codegen| codegen.emit()) {
            Ok(_) => assert!(valid),
            Err(error) => {
                assert!(!valid);
                assert_eq!(error.source, support::source());
                assert!(
                    error
                        .message
                        .contains("at most 28 buffer and value parameters")
                );
            }
        }
    }
}

/// A kernel that loads and stores under `n < 16`, then stores in a loop; either side can be
/// left without effects.
fn gated_kernel(guarded: bool, looped: bool) -> Module {
    let buffer = |access| Type::Buffer {
        element: Element::Scalar(Scalar::U32),
        access,
        address_space: AddressSpace::Device,
    };
    let store = |index| Expression::BufferStore {
        buffer: local(2),
        index: local(index),
        value: local(index),
    };
    let guarded = if guarded {
        vec![
            (
                5,
                Expression::BufferLoad {
                    buffer: local(1),
                    index: local(3),
                },
            ),
            (6, store(5)),
        ]
    } else {
        vec![(5, Expression::Use(local(3)))]
    };
    let mut looped = if looped { vec![(6, store(7))] } else { vec![] };
    looped.push((7, Expression::Binary(BinaryOp::Add, local(7), uint(1))));
    let mut module = module(
        3,
        vec![
            Type::Unit,
            buffer(Access::Read),
            buffer(Access::Write),
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::Bool),
            Type::Scalar(Scalar::U32),
            Type::Unit,
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::Bool),
        ],
        vec![
            block(
                vec![(4, Expression::Binary(BinaryOp::Lt, local(3), uint(16)))],
                Terminator::Branch {
                    condition: local(4),
                    then_block: 1,
                    else_block: 2,
                },
            ),
            block(guarded, Terminator::Return),
            block(vec![(7, Expression::Use(uint(0)))], Terminator::Goto(3)),
            block(
                vec![(8, Expression::Binary(BinaryOp::Lt, local(7), local(3)))],
                Terminator::Branch {
                    condition: local(8),
                    then_block: 4,
                    else_block: 5,
                },
            ),
            block(looped, Terminator::Goto(3)),
            block(vec![], Terminator::Return),
        ],
    );
    module.functions[0].kernel = true;
    module.functions[0].name = "gated".into();
    module
}

#[test]
fn kernels_test_their_gate_once_before_every_effect() {
    for (guarded, looped, before) in [(true, true, "if (v4) {"), (false, true, "while (true) {")] {
        let lines = gated_lines(guarded, looped);
        let gates = gates(&lines);
        assert_eq!(gates.len(), 1, "{lines:#?}");
        assert_eq!(lines[gates[0] + 1], before, "{lines:#?}");
    }
    let lines = gated_lines(true, false);
    let gates = gates(&lines);
    assert_eq!(gates.len(), 1, "{lines:#?}");
    assert_eq!(lines[gates[0] - 1], "if (v4) {", "{lines:#?}");
}

fn gated_lines(guarded: bool, looped: bool) -> Vec<String> {
    let msl = metal_oxide_codegen::Codegen::new(&gated_kernel(guarded, looped))
        .and_then(|codegen| codegen.emit())
        .unwrap();
    msl[msl.find("kernel void gated(").unwrap()..]
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.starts_with("#line "))
        .collect()
}

fn gates(lines: &[String]) -> Vec<usize> {
    lines
        .iter()
        .enumerate()
        .filter(|(_, line)| *line == "if (*metal_oxide_gate != 0u) { return; }")
        .map(|(index, _)| index)
        .collect()
}
