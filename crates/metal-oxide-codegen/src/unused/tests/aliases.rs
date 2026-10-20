use super::*;

#[test]
fn slices_offsets_and_copied_closure_captures_keep_their_storage_live() {
    let mut types = TypeTable::default();
    let array = array(&mut types);
    let array_pointer = pointer(&mut types, array, false);
    let slice = pointer(&mut types, U32, true);
    let thin = pointer(&mut types, U32, false);
    let closure = types.intern(Aggregate::record("Closure", vec![slice]));
    let used = used(
        types,
        vec![
            function(
                0,
                vec![
                    U32,
                    array,
                    array_pointer,
                    slice,
                    slice,
                    thin,
                    thin,
                    slice,
                    closure,
                    closure,
                    USIZE,
                    USIZE,
                    USIZE,
                    U32,
                ],
                vec![block(
                    vec![
                        assign(1, array_value(array, 7, 9)),
                        assign(2, reference(Place::local(1))),
                        assign(3, Expression::Slice(Operand::local(2))),
                        assign(10, index(0)),
                        assign(11, index(1)),
                        assign(12, index(2)),
                        assign(
                            4,
                            Expression::Subslice {
                                slice: Operand::local(3),
                                start: Operand::local(10),
                                end: Operand::local(12),
                            },
                        ),
                        assign(5, Expression::PointerCast(Operand::local(4), thin)),
                        assign(
                            6,
                            Expression::Binary(
                                BinaryOp::Offset,
                                Operand::local(5),
                                Operand::local(11),
                            ),
                        ),
                        assign(
                            7,
                            Expression::SliceFromParts {
                                data: Operand::local(6),
                                length: Operand::local(11),
                            },
                        ),
                        assign(8, aggregate(closure, vec![Operand::local(7)])),
                        assign(9, value(8)),
                        assign(
                            13,
                            Expression::Call {
                                function: 1,
                                arguments: vec![Operand::local(9)],
                            },
                        ),
                        store(projected(1, vec![Projection::Index(10)]), uint(100)),
                        assign(0, value(13)),
                    ],
                    Terminator::Return,
                )],
            ),
            function(
                1,
                vec![U32, closure, USIZE, USIZE, BOOL],
                vec![
                    block(
                        vec![
                            assign(2, index(0)),
                            assign(
                                3,
                                Expression::Len(Operand::Place(projected(
                                    1,
                                    vec![Projection::Field(0)],
                                ))),
                            ),
                            assign(
                                4,
                                Expression::Binary(
                                    BinaryOp::Lt,
                                    Operand::local(2),
                                    Operand::local(3),
                                ),
                            ),
                        ],
                        Terminator::Assert {
                            condition: Operand::local(4),
                            expected: true,
                            enabled: true,
                            target: 1,
                            check: Check::Bounds,
                        },
                    ),
                    block(
                        vec![assign(
                            0,
                            Expression::Use(Operand::Place(projected(
                                1,
                                vec![
                                    Projection::Field(0),
                                    Projection::Deref,
                                    Projection::Index(2),
                                ],
                            ))),
                        )],
                        Terminator::Return,
                    ),
                ],
            ),
        ],
    );
    assert!(used.statements[0][..13].iter().all(|&keep| keep));
    assert!(!used.statement(0, 13));
    assert!(used.statement(0, 14));
}

#[test]
fn a_helper_read_only_keeps_the_referent_of_its_argument_live() {
    let mut types = TypeTable::default();
    let pointer = pointer(&mut types, U32, false);
    let used = used(
        types,
        vec![
            function(
                0,
                vec![U32, U32, U32, pointer, pointer, U32],
                vec![block(
                    vec![
                        assign(1, uint(1)),
                        assign(2, uint(2)),
                        assign(3, reference(Place::local(1))),
                        assign(4, reference(Place::local(2))),
                        assign(
                            5,
                            Expression::Call {
                                function: 1,
                                arguments: vec![Operand::local(3)],
                            },
                        ),
                        assign(1, uint(3)),
                        assign(
                            0,
                            Expression::Call {
                                function: 1,
                                arguments: vec![Operand::local(4)],
                            },
                        ),
                    ],
                    Terminator::Return,
                )],
            ),
            function(
                1,
                vec![U32, pointer],
                vec![block(
                    vec![assign(
                        0,
                        Expression::Use(Operand::Place(projected(1, vec![Projection::Deref]))),
                    )],
                    Terminator::Return,
                )],
            ),
        ],
    );
    assert_eq!(
        used.statements[0],
        [true, true, true, true, true, false, true]
    );
}

#[test]
fn helpers_can_replace_a_borrowed_pointer() {
    let mut types = TypeTable::default();
    let p = pointer(&mut types, U32, false);
    let pp = pointer(&mut types, p, false);
    let used = used(
        types,
        vec![
            function(
                0,
                vec![U32, U32, U32, p, p, pp, Type::Unit],
                vec![block(
                    vec![
                        assign(1, uint(7)),
                        assign(2, uint(9)),
                        assign(3, reference(Place::local(1))),
                        assign(4, reference(Place::local(2))),
                        assign(5, reference(Place::local(3))),
                        assign(
                            6,
                            Expression::Call {
                                function: 1,
                                arguments: vec![Operand::local(5), Operand::local(4)],
                            },
                        ),
                        assign(2, uint(11)),
                        assign(
                            0,
                            Expression::Use(Operand::Place(projected(3, vec![Projection::Deref]))),
                        ),
                    ],
                    Terminator::Return,
                )],
            ),
            function(
                2,
                vec![Type::Unit, pp, p],
                vec![block(
                    vec![store(projected(1, vec![Projection::Deref]), value(2))],
                    Terminator::Return,
                )],
            ),
        ],
    );
    assert!(used.statements[0].iter().all(|&keep| keep));
}

#[test]
fn nested_reborrows_preserve_pointer_values_and_the_final_referent() {
    let mut types = TypeTable::default();
    let p = pointer(&mut types, U32, false);
    let pp = pointer(&mut types, p, false);
    let used = used(
        types,
        vec![function(
            0,
            vec![U32, U32, p, pp, p],
            vec![block(
                vec![
                    assign(1, uint(7)),
                    assign(2, reference(Place::local(1))),
                    assign(3, reference(Place::local(2))),
                    assign(
                        4,
                        reference(projected(3, vec![Projection::Deref, Projection::Deref])),
                    ),
                    assign(
                        0,
                        Expression::Use(Operand::Place(projected(4, vec![Projection::Deref]))),
                    ),
                ],
                Terminator::Return,
            )],
        )],
    );
    assert!(used.statements[0].iter().all(|&keep| keep));
}

#[test]
fn a_pointer_returned_by_a_helper_can_observe_later_writes() {
    let mut types = TypeTable::default();
    let p = pointer(&mut types, U32, false);
    let used = used(
        types,
        vec![
            function(
                0,
                vec![U32, U32, p, p],
                vec![block(
                    vec![
                        assign(1, uint(7)),
                        assign(2, reference(Place::local(1))),
                        assign(
                            3,
                            Expression::Call {
                                function: 1,
                                arguments: vec![Operand::local(2)],
                            },
                        ),
                        assign(1, uint(9)),
                        assign(
                            0,
                            Expression::Use(Operand::Place(projected(3, vec![Projection::Deref]))),
                        ),
                    ],
                    Terminator::Return,
                )],
            ),
            function(
                1,
                vec![p, p],
                vec![block(vec![assign(0, value(1))], Terminator::Return)],
            ),
        ],
    );
    assert!(used.statements[0].iter().all(|&keep| keep));
}

#[test]
fn helpers_can_replace_a_reference_inside_a_borrowed_aggregate() {
    let mut types = TypeTable::default();
    let p = pointer(&mut types, U32, false);
    let tuple = types.intern(Aggregate::Tuple(vec![p]));
    let tuple_pointer = pointer(&mut types, tuple, false);
    let used = used(
        types,
        vec![
            function(
                0,
                vec![U32, U32, U32, p, p, tuple, tuple_pointer, Type::Unit],
                vec![block(
                    vec![
                        assign(1, uint(7)),
                        assign(2, uint(9)),
                        assign(3, reference(Place::local(1))),
                        assign(4, reference(Place::local(2))),
                        assign(5, aggregate(tuple, vec![Operand::local(3)])),
                        assign(6, reference(Place::local(5))),
                        assign(
                            7,
                            Expression::Call {
                                function: 1,
                                arguments: vec![Operand::local(6), Operand::local(4)],
                            },
                        ),
                        assign(2, uint(11)),
                        assign(
                            0,
                            Expression::Use(Operand::Place(projected(
                                5,
                                vec![Projection::Field(0), Projection::Deref],
                            ))),
                        ),
                    ],
                    Terminator::Return,
                )],
            ),
            function(
                2,
                vec![Type::Unit, tuple_pointer, p],
                vec![block(
                    vec![store(
                        projected(1, vec![Projection::Deref, Projection::Field(0)]),
                        value(2),
                    )],
                    Terminator::Return,
                )],
            ),
        ],
    );
    assert!(used.statements[0].iter().all(|&keep| keep));
}
