use crate::{Access, Layout, ParameterType, Scalar};

#[test]
fn layouts_print_as_rust_types() {
    let f32 = Layout::scalar(Scalar::F32);
    let pair = Layout::tuple(vec![f32.clone(), Layout::scalar(Scalar::BF16)]).unwrap();
    let single = Layout::tuple(vec![Layout::scalar(Scalar::I64)]).unwrap();
    let array = Layout::array(Layout::scalar(Scalar::U8), 4).unwrap();
    let record = Layout::record("Particle", vec![("mass".into(), f32.clone())]).unwrap();
    assert_eq!(f32.to_string(), "f32");
    assert_eq!(pair.to_string(), "(f32, BF16)");
    assert_eq!(single.to_string(), "(i64,)");
    assert_eq!(array.to_string(), "[u8; 4]");
    assert_eq!(record.to_string(), "Particle");
}

#[test]
fn parameter_types_name_their_access_and_element() {
    let element = Layout::scalar(Scalar::U32);
    let buffer = |access| ParameterType::Buffer {
        element: element.clone(),
        stride: 4,
        access,
    };
    assert_eq!(buffer(Access::Read).to_string(), "read buffer of u32");
    assert_eq!(
        buffer(Access::ReadWrite).to_string(),
        "read-write buffer of u32"
    );
    assert_eq!(buffer(Access::Atomic).to_string(), "atomic buffer of u32");
    let value = ParameterType::Value {
        layout: Layout::scalar(Scalar::F16),
    };
    assert_eq!(value.to_string(), "value of F16");
    let shared = ParameterType::Threadgroup {
        element: element.clone(),
        stride: 4,
    };
    assert_eq!(shared.to_string(), "threadgroup buffer of u32");
}
