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
