mod support;

use metal_oxide_ir::*;
use support::*;

#[test]
fn direct_ir_division_requires_checks_even_without_an_assert() {
    for divisor in [local(2), Operand::Constant(Constant::I32(-1))] {
        let module = module(
            2,
            vec![Type::Scalar(Scalar::I32); 3],
            vec![block(
                vec![(0, Expression::Binary(BinaryOp::Div, local(1), divisor))],
                Terminator::Return,
            )],
        );
        let error = metal_oxide_codegen::Codegen::new(&module).unwrap_err();
        assert!(error.message.contains("proved"), "{error}");
        assert_eq!(validate(&module).unwrap_err(), error);
    }
}

#[test]
fn dynamic_switch_discriminants_require_a_bounds_proof() {
    let mut types = TypeTable::default();
    let array = types.intern(Aggregate::Array {
        element: Type::Scalar(Scalar::U32),
        length: 4,
    });
    let mut module = module(
        2,
        vec![Type::Unit, array, Type::Scalar(Scalar::Usize)],
        vec![
            block(
                vec![],
                Terminator::Switch {
                    discriminant: Operand::Place(Place {
                        local: 1,
                        projection: vec![Projection::Index(2)],
                    }),
                    cases: vec![(Constant::U32(0), 1)],
                    otherwise: 1,
                },
            ),
            block(vec![], Terminator::Return),
        ],
    );
    module.types = types;
    let error = metal_oxide_codegen::Codegen::new(&module).unwrap_err();
    assert!(error.message.contains("bounds"), "{error}");
    assert_eq!(validate(&module).unwrap_err(), error);
}

#[test]
fn usize_casts_and_shifts_preserve_all_sixty_four_bits() {
    let module = module(
        1,
        vec![Type::Scalar(Scalar::Usize); 2],
        vec![block(
            vec![(0, Expression::Binary(BinaryOp::Shl, local(1), uint(32)))],
            Terminator::Return,
        )],
    );
    assert_eq!(
        execute(&module, "std::cout << metal_oxide_fn_0(1ul, ctx);"),
        "4294967296"
    );
}

#[test]
fn half_storage_does_not_implicitly_enable_half_arithmetic() {
    for (result, expression) in [
        (
            Scalar::F16,
            Expression::Binary(BinaryOp::Add, local(1), local(1)),
        ),
        (Scalar::I32, Expression::Cast(local(1), Scalar::I32)),
        (
            Scalar::F32,
            Expression::Math {
                op: MathOp::Sqrt,
                arguments: vec![local(1)],
            },
        ),
    ] {
        let module = module(
            1,
            vec![Type::Scalar(result), Type::Scalar(Scalar::F16)],
            vec![block(vec![(0, expression)], Terminator::Return)],
        );
        assert!(metal_oxide_codegen::Codegen::new(&module).is_err());
    }
}
