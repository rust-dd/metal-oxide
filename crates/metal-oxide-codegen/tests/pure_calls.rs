mod support;

use metal_oxide_ir::*;
use support::{block, local, module, uint};

#[test]
fn shared_return_can_duplicate_a_pure_helper_call() {
    let mut module = module(
        1,
        vec![Type::Scalar(Scalar::U32), Type::Scalar(Scalar::Bool)],
        vec![
            block(
                vec![],
                Terminator::Branch {
                    condition: local(1),
                    then_block: 1,
                    else_block: 2,
                },
            ),
            block(vec![], Terminator::Goto(3)),
            block(vec![], Terminator::Goto(3)),
            block(
                vec![(
                    0,
                    Expression::Call {
                        function: 1,
                        arguments: vec![uint(4)],
                    },
                )],
                Terminator::Return,
            ),
        ],
    );
    let helper = support::module(
        1,
        vec![Type::Scalar(Scalar::U32); 2],
        vec![block(
            vec![(0, Expression::Use(local(1)))],
            Terminator::Return,
        )],
    )
    .functions
    .remove(0);
    module.functions.push(helper);
    assert_eq!(
        support::execute(
            &module,
            "std::cout << metal_oxide_fn_0(true, ctx) << ',' << metal_oxide_fn_0(false, ctx);"
        ),
        "4,4"
    );
}
