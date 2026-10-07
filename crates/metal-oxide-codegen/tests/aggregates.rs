mod support;

use metal_oxide_ir::*;
use support::{block, local, module, uint};

fn id(ty: Type) -> usize {
    let Type::Aggregate(id) = ty else {
        panic!("expected aggregate type")
    };
    id
}

fn field(local: usize, field: u32) -> Operand {
    Operand::Place(Place {
        local,
        projection: vec![Projection::Field(field)],
    })
}

fn construct(ty: Type, fields: Vec<Operand>) -> Expression {
    Expression::Aggregate { ty: id(ty), fields }
}

fn update(aggregate: Operand, field: u32, value: Operand) -> Expression {
    Expression::AggregateUpdate {
        aggregate,
        field,
        value,
    }
}

#[test]
fn nested_record_tuple_and_array_copies_survive_helper_calls() {
    let mut types = TypeTable::default();
    let array = types.intern(Aggregate::Array {
        element: Type::Scalar(Scalar::U32),
        length: 3,
    });
    let tuple = types.intern(Aggregate::Tuple(vec![Type::Scalar(Scalar::U16), array]));
    let record = types.intern(Aggregate::record(
        "Record",
        vec![Type::Scalar(Scalar::U32), tuple],
    ));
    let mut output = module(
        1,
        vec![record, record, tuple, array, array, tuple],
        vec![block(
            vec![
                (2, Expression::Use(field(1, 1))),
                (3, Expression::Use(field(2, 1))),
                (4, update(local(3), 1, uint(99))),
                (5, update(local(2), 1, local(4))),
                (0, update(local(1), 1, local(5))),
            ],
            Terminator::Return,
        )],
    );
    let constructor = module(
        0,
        vec![record, array, tuple],
        vec![block(
            vec![
                (1, construct(array, vec![uint(10), uint(20), uint(30)])),
                (
                    2,
                    construct(tuple, vec![Operand::Constant(Constant::U16(7)), local(1)]),
                ),
                (0, construct(record, vec![uint(5), local(2)])),
            ],
            Terminator::Return,
        )],
    )
    .functions
    .remove(0);
    let caller = module(
        0,
        vec![record, record],
        vec![block(
            vec![
                (
                    1,
                    Expression::Call {
                        function: 1,
                        arguments: vec![],
                    },
                ),
                (
                    0,
                    Expression::Call {
                        function: 0,
                        arguments: vec![local(1)],
                    },
                ),
            ],
            Terminator::Return,
        )],
    )
    .functions
    .remove(0);
    output.types = types;
    output.functions.extend([constructor, caller]);
    assert_eq!(
        support::execute(
            &output,
            "auto original = metal_oxide_fn_1(ctx); auto changed = metal_oxide_fn_2(ctx); std::cout << original.f0 << ',' << original.f1.f0 << ',' << original.f1.f1.elements[1] << ',' << changed.f0 << ',' << changed.f1.f0 << ',' << changed.f1.f1.elements[1] << ',' << changed.f1.f1.elements[0] << ',' << changed.f1.f1.elements[2];"
        ),
        "5,7,20,5,7,99,10,30"
    );
}

#[test]
fn arrays_of_records_and_nested_arrays_preserve_construction_order() {
    let mut types = TypeTable::default();
    let record = types.intern(Aggregate::record(
        "Record",
        vec![Type::Scalar(Scalar::F32), Type::Scalar(Scalar::U8)],
    ));
    let inner = types.intern(Aggregate::Array {
        element: record,
        length: 2,
    });
    let outer = types.intern(Aggregate::Array {
        element: inner,
        length: 2,
    });
    let value = |float: f32, byte| {
        construct(
            record,
            vec![
                Operand::Constant(Constant::F32(float.to_bits())),
                Operand::Constant(Constant::U8(byte)),
            ],
        )
    };
    let mut output = module(
        0,
        vec![outer, record, record, inner, inner],
        vec![block(
            vec![
                (1, value(1.0, 2)),
                (2, value(3.0, 4)),
                (3, construct(inner, vec![local(1), local(2)])),
                (4, construct(inner, vec![local(2), local(1)])),
                (0, construct(outer, vec![local(3), local(4)])),
            ],
            Terminator::Return,
        )],
    );
    output.types = types;
    assert_eq!(
        support::execute(
            &output,
            "auto value = metal_oxide_fn_0(ctx); std::cout << value.elements[0].elements[1].f0 << ',' << int(value.elements[1].elements[0].f1) << ',' << value.elements[1].elements[1].f0;"
        ),
        "3,4,1"
    );
}

#[test]
fn replacement_reads_the_old_destination_before_it_is_overwritten() {
    let mut types = TypeTable::default();
    let record = types.intern(Aggregate::record(
        "Record",
        vec![Type::Scalar(Scalar::U32); 2],
    ));
    let mut output = module(
        2,
        vec![record; 3],
        vec![block(
            vec![
                (0, Expression::Use(local(1))),
                (0, update(local(2), 0, field(0, 1))),
            ],
            Terminator::Return,
        )],
    );
    output.types = types;
    assert_eq!(
        support::execute(
            &output,
            "auto value = metal_oxide_fn_0({1u, 5u}, {10u, 20u}, ctx); std::cout << value.f0 << ',' << value.f1;"
        ),
        "5,20"
    );
}

#[test]
fn checked_types_inside_aggregates_have_their_msl_definition() {
    let mut types = TypeTable::default();
    let tuple = types.intern(Aggregate::Tuple(vec![Type::Checked(Scalar::U16)]));
    let mut output = module(
        1,
        vec![tuple; 2],
        vec![block(
            vec![(0, Expression::Use(local(1)))],
            Terminator::Return,
        )],
    );
    output.types = types;
    assert_eq!(
        support::execute(
            &output,
            "auto value = metal_oxide_fn_0({{ushort(42u), false}}, ctx); std::cout << value.f0.value << ',' << value.f0.overflow;"
        ),
        "42,0"
    );
}

#[test]
fn array_update_source_size_is_bounded_independently_of_length() {
    for length in [1, 100_000] {
        let mut types = TypeTable::default();
        let array = types.intern(Aggregate::Array {
            element: Type::Scalar(Scalar::U32),
            length,
        });
        let mut output = module(
            1,
            vec![array; 2],
            vec![block(
                vec![(0, update(local(1), 0, uint(9)))],
                Terminator::Return,
            )],
        );
        output.types = types;
        let msl = metal_oxide_codegen::Codegen::new(&output)
            .and_then(|codegen| codegen.emit())
            .unwrap();
        assert!(msl.contains(&format!("elements[{length}]")));
        assert!(
            msl.len() < 2_000,
            "expanded array update to {} bytes",
            msl.len()
        );
    }
}
