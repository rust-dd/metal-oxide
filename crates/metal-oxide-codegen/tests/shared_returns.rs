mod support;

use metal_oxide_ir::*;
use support::{block, local, module, uint};

#[test]
fn many_branches_emit_their_shared_return_once() {
    let mut blocks = vec![block(
        vec![],
        Terminator::Switch {
            discriminant: local(1),
            cases: (0..9)
                .map(|value| (Constant::U32(value), value as usize + 1))
                .collect(),
            otherwise: 10,
        },
    )];
    blocks.extend((0..10).map(|value| {
        block(
            vec![(0, Expression::Use(uint(value)))],
            Terminator::Goto(11),
        )
    }));
    blocks.push(block(
        vec![(0, Expression::Binary(BinaryOp::Add, local(0), uint(100)))],
        Terminator::Return,
    ));
    let module = module(1, vec![Type::Scalar(Scalar::U32); 2], blocks);
    let msl = metal_oxide_codegen::Codegen::new(&module)
        .unwrap()
        .emit()
        .unwrap();
    assert_eq!(msl.matches("return v0;").count(), 1, "{msl}");
    assert_eq!(
        support::execute(
            &module,
            "for (uint i = 0; i < 11; ++i) std::cout << metal_oxide_fn_0(i, ctx) << ',';",
        ),
        "100,101,102,103,104,105,106,107,108,109,109,",
    );
}

#[test]
fn a_shared_terminal_barrier_is_emitted_once() {
    let mut module = module(
        1,
        vec![Type::Unit, Type::Scalar(Scalar::U32), Type::Unit],
        vec![
            block(
                vec![],
                Terminator::Switch {
                    discriminant: local(1),
                    cases: vec![(Constant::U32(0), 1), (Constant::U32(1), 2)],
                    otherwise: 3,
                },
            ),
            block(vec![], Terminator::Goto(4)),
            block(vec![], Terminator::Goto(4)),
            block(vec![], Terminator::Goto(4)),
            block(
                vec![(2, Expression::ThreadgroupBarrier)],
                Terminator::Return,
            ),
        ],
    );
    module.functions[0].kernel = true;
    let msl = metal_oxide_codegen::Codegen::new(&module)
        .unwrap()
        .emit()
        .unwrap();
    assert_eq!(msl.matches("threadgroup_barrier(").count(), 1, "{msl}");
}

#[test]
fn short_circuit_branches_join_before_a_cooperative_tail() {
    for indirect in [false, true] {
        let mut module = module(
            3,
            vec![
                Type::Unit,
                Type::Scalar(Scalar::U32),
                Type::Scalar(Scalar::U32),
                Type::Scalar(Scalar::U32),
                Type::Unit,
            ],
            vec![
                block(
                    vec![],
                    Terminator::Switch {
                        discriminant: local(1),
                        cases: vec![(Constant::U32(0), 1)],
                        otherwise: 2,
                    },
                ),
                block(vec![], Terminator::Goto(6)),
                block(
                    vec![],
                    Terminator::Switch {
                        discriminant: local(2),
                        cases: vec![(Constant::U32(0), 3)],
                        otherwise: 4,
                    },
                ),
                block(
                    vec![],
                    Terminator::Switch {
                        discriminant: local(3),
                        cases: vec![(Constant::U32(0), 5)],
                        otherwise: 4,
                    },
                ),
                block(
                    vec![],
                    Terminator::Switch {
                        discriminant: local(1),
                        cases: vec![(Constant::U32(0), 5)],
                        otherwise: 6,
                    },
                ),
                block(vec![], Terminator::Goto(6)),
                block(
                    vec![(4, Expression::ThreadgroupBarrier)],
                    Terminator::Return,
                ),
            ],
        );
        module.functions[0].kernel = true;
        if indirect {
            let helper = support::module(
                0,
                vec![Type::Unit; 2],
                vec![block(
                    vec![(1, Expression::ThreadgroupBarrier)],
                    Terminator::Return,
                )],
            )
            .functions
            .remove(0);
            module.functions.push(helper);
            module.functions[0].blocks[6].statements[0].value = Expression::Call {
                function: 1,
                arguments: vec![],
            };
        }
        let msl = metal_oxide_codegen::Codegen::new(&module)
            .unwrap()
            .emit()
            .unwrap();
        assert_eq!(msl.matches("threadgroup_barrier(").count(), 1, "{msl}");
    }
}

#[test]
fn nested_terminal_joins_preserve_the_enclosing_barrier_continuation() {
    let mut module = module(
        3,
        vec![
            Type::Unit,
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::U32),
            Type::Unit,
        ],
        vec![
            block(
                vec![],
                Terminator::Switch {
                    discriminant: local(1),
                    cases: vec![(Constant::U32(0), 1)],
                    otherwise: 2,
                },
            ),
            block(
                vec![],
                Terminator::Switch {
                    discriminant: local(2),
                    cases: vec![(Constant::U32(0), 3)],
                    otherwise: 4,
                },
            ),
            block(
                vec![],
                Terminator::Switch {
                    discriminant: local(3),
                    cases: vec![(Constant::U32(0), 3)],
                    otherwise: 5,
                },
            ),
            block(
                vec![(4, Expression::ThreadgroupBarrier)],
                Terminator::Goto(4),
            ),
            block(vec![], Terminator::Return),
            block(vec![], Terminator::Return),
        ],
    );
    module.functions[0].kernel = true;
    let msl = metal_oxide_codegen::Codegen::new(&module)
        .unwrap()
        .emit()
        .unwrap();
    assert_eq!(msl.matches("threadgroup_barrier(").count(), 1, "{msl}");

    // The portable shim has no barrier; count visits on the identical control-flow graph.
    let function = &mut module.functions[0];
    function.kernel = false;
    function.locals[0] = Type::Scalar(Scalar::U32);
    function.blocks[0].statements =
        block(vec![(0, Expression::Use(uint(0)))], Terminator::Return).statements;
    function.blocks[3] = block(
        vec![(0, Expression::Binary(BinaryOp::Add, local(0), uint(1)))],
        Terminator::Goto(4),
    );
    for (index, value) in [(4, 100), (5, 200)] {
        function.blocks[index] = block(
            vec![(0, Expression::Binary(BinaryOp::Add, local(0), uint(value)))],
            Terminator::Return,
        );
    }
    assert_eq!(
        support::execute(
            &module,
            "for (uint a = 0; a < 2; ++a) for (uint b = 0; b < 2; ++b) \
             for (uint c = 0; c < 2; ++c) std::cout << metal_oxide_fn_0(a, b, c, ctx) << ',';",
        ),
        "101,101,100,100,101,200,101,200,",
    );
}

#[test]
fn loop_exits_join_before_the_code_they_share() {
    let condition = |op, a, b| (4, Expression::Binary(op, a, b));
    let branch = |then_block, else_block| Terminator::Branch {
        condition: local(4),
        then_block,
        else_block,
    };
    // Exit 5 returns early; exits 6 and 7 continue at 8, which holds a barrier.
    let mut module = module(
        2,
        vec![
            Type::Unit,
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::U32),
            Type::Scalar(Scalar::Bool),
            Type::Unit,
        ],
        vec![
            block(vec![(3, Expression::Use(uint(0)))], Terminator::Goto(1)),
            block(
                vec![condition(BinaryOp::Lt, local(3), local(1))],
                branch(2, 6),
            ),
            block(
                vec![condition(BinaryOp::Eq, local(3), local(2))],
                branch(5, 3),
            ),
            block(
                vec![condition(BinaryOp::Eq, local(3), uint(5))],
                branch(7, 4),
            ),
            block(
                vec![(3, Expression::Binary(BinaryOp::Add, local(3), uint(1)))],
                Terminator::Goto(1),
            ),
            block(vec![], Terminator::Return),
            block(vec![], Terminator::Goto(8)),
            block(vec![], Terminator::Goto(8)),
            block(
                vec![(5, Expression::ThreadgroupBarrier)],
                Terminator::Goto(9),
            ),
            block(vec![], Terminator::Return),
        ],
    );
    module.functions[0].kernel = true;
    let msl = metal_oxide_codegen::Codegen::new(&module)
        .unwrap()
        .emit()
        .unwrap();
    assert_eq!(msl.matches("threadgroup_barrier(").count(), 1, "{msl}");

    let function = &mut module.functions[0];
    function.kernel = false;
    function.locals[0] = Type::Scalar(Scalar::U32);
    function.blocks[5] = block(vec![(0, Expression::Use(uint(7)))], Terminator::Return);
    function.blocks[8] = block(
        vec![(0, Expression::Binary(BinaryOp::Add, local(3), uint(100)))],
        Terminator::Goto(9),
    );
    let msl = metal_oxide_codegen::Codegen::new(&module)
        .unwrap()
        .emit()
        .unwrap();
    assert_eq!(msl.matches("100u").count(), 1, "{msl}");
    assert_eq!(
        support::execute(
            &module,
            "for (auto [n, a] : {std::pair{3u, 9u}, {9u, 2u}, {9u, 7u}, {0u, 0u}}) \
             std::cout << metal_oxide_fn_0(n, a, ctx) << ',';",
        ),
        "103,7,105,100,",
    );
}
