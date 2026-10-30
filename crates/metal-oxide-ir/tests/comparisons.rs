mod support;

use metal_oxide_ir::*;
use support::{assign, block, function, module};

fn comparison(left: Type, right: Type) -> Module {
    let mut function = function(
        "compare",
        false,
        vec![Type::Scalar(Scalar::I8), left, right],
        vec![block(
            vec![assign(
                0,
                Expression::Binary(BinaryOp::Cmp, Operand::local(1), Operand::local(2)),
            )],
            Terminator::Return,
        )],
    );
    function.parameters = 2;
    module(vec![function])
}

#[test]
fn three_way_comparison_returns_signed_byte_for_integer_and_boolean_operands() {
    for scalar in [
        Scalar::Bool,
        Scalar::I8,
        Scalar::U8,
        Scalar::I16,
        Scalar::U16,
        Scalar::I32,
        Scalar::U32,
        Scalar::U64,
        Scalar::I64,
        Scalar::Usize,
        Scalar::Isize,
    ] {
        let ty = Type::Scalar(scalar);
        validate(&comparison(ty, ty)).unwrap();
    }
}

#[test]
fn three_way_comparison_rejects_floats_and_mismatched_operands() {
    for (left, right) in [
        (Scalar::F32, Scalar::F32),
        (Scalar::U32, Scalar::I32),
        (Scalar::U8, Scalar::U32),
        (Scalar::Bool, Scalar::U32),
    ] {
        let error = validate(&comparison(Type::Scalar(left), Type::Scalar(right))).unwrap_err();
        assert_eq!(error.message, "invalid binary operation types");
    }
}

#[test]
fn three_way_comparison_rejects_pointers() {
    let mut module = comparison(Type::Unit, Type::Unit);
    let pointer = module.types.intern_pointer(Pointer {
        target: Type::Scalar(Scalar::U32),
        mutable: false,
        slice: false,
        space: AddressSpace::Device,
    });
    module.functions[0].locals[1..].fill(pointer);
    assert_eq!(
        validate(&module).unwrap_err().message,
        "invalid binary operation types"
    );
}
