use metal_oxide_artifact::{FieldLayout, Layout, LayoutKind, Scalar};

fn particle() -> Layout {
    Layout::record(
        "Particle",
        vec![
            ("tag".into(), Layout::scalar(Scalar::U8)),
            ("flags".into(), Layout::scalar(Scalar::U16)),
            ("id".into(), Layout::scalar(Scalar::U32)),
            (
                "position".into(),
                Layout::tuple(vec![Layout::scalar(Scalar::F32); 3]).unwrap(),
            ),
            (
                "weights".into(),
                Layout::array(Layout::scalar(Scalar::F32), 2).unwrap(),
            ),
        ],
    )
    .unwrap()
}

#[test]
fn mixed_fields_and_nested_values_have_canonical_offsets_and_stride() {
    let value = particle();
    assert_eq!((value.size, value.alignment), (28, 4));
    let LayoutKind::Record { fields, .. } = &value.kind else {
        panic!()
    };
    assert_eq!(
        fields.iter().map(|field| field.offset).collect::<Vec<_>>(),
        [0, 2, 4, 8, 20]
    );
    let array = Layout::array(value, 3).unwrap();
    assert_eq!((array.size, array.alignment), (84, 4));
    let LayoutKind::Array { stride, .. } = array.kind else {
        panic!()
    };
    assert_eq!(stride, 28);
}

#[test]
fn explicit_padding_and_tail_alignment_are_verified() {
    let value = Layout::record(
        "Padding",
        vec![
            ("x".into(), Layout::scalar(Scalar::U32)),
            ("tag".into(), Layout::scalar(Scalar::U8)),
        ],
    )
    .unwrap();
    assert_eq!((value.size, value.alignment), (8, 4));
    for wrong in [
        Layout {
            size: 5,
            ..value.clone()
        },
        Layout {
            alignment: 1,
            ..value.clone()
        },
    ] {
        assert!(wrong.validate().is_err());
    }
    let mut wrong = value;
    let LayoutKind::Record { fields, .. } = &mut wrong.kind else {
        panic!()
    };
    fields[1].offset = 3;
    assert!(wrong.validate().is_err());
}

#[test]
fn layouts_reject_empty_shapes_duplicate_fields_and_overflow() {
    assert!(Layout::record("Empty", vec![]).is_err());
    assert!(Layout::tuple(vec![]).is_err());
    assert!(Layout::array(Layout::scalar(Scalar::U8), 0).is_err());
    assert!(
        Layout::record(
            "Duplicate",
            vec![("x".into(), Layout::scalar(Scalar::U8)); 2]
        )
        .is_err()
    );
    let huge = Layout::array(Layout::scalar(Scalar::U32), u32::MAX).unwrap();
    assert!(Layout::array(huge, u32::MAX).is_err());
    let wrong = Layout {
        size: 8,
        alignment: 4,
        kind: LayoutKind::Record {
            name: "Wrong".into(),
            fields: vec![FieldLayout {
                name: "x".into(),
                offset: 0,
                layout: Layout {
                    size: 3,
                    ..Layout::scalar(Scalar::U32)
                },
            }],
        },
    };
    assert!(wrong.validate().is_err());
}

#[test]
fn nested_layout_json_rejects_invalid_stride_and_unknown_fields() {
    let value = Layout::array(particle(), 2).unwrap();
    let json = serde_json::to_string(&value).unwrap();
    let decoded: Layout = serde_json::from_str(&json).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded, value);
    let changed: Layout =
        serde_json::from_str(&json.replace("\"stride\":28", "\"stride\":27")).unwrap();
    assert!(changed.validate().is_err());
    assert!(
        serde_json::from_str::<Layout>(
            &json.replace("\"size\":56", "\"size\":56,\"pointer\":true")
        )
        .is_err()
    );
}
