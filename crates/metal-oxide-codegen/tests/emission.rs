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
fn enabled_assertion_is_not_silently_discarded() {
    let module = assertion(true);
    let error = metal_oxide_codegen::Codegen::new(&module)
        .and_then(|codegen| codegen.emit())
        .unwrap_err();
    assert_eq!(error.source, support::source());
    assert!(error.message.contains("assertion"));
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
                    message: "overflow".into(),
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
    for (parameters, kernel, valid) in [(31, true, true), (32, true, false), (32, false, true)] {
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
                assert!(error.message.contains("31 kernel parameters"));
            }
        }
    }
}
