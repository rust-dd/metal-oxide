use metal_oxide::GpuValue;
use metal_oxide_artifact::{Layout, Scalar};

#[test]
fn owned_values_use_canonical_offsets_and_zero_padding() {
    type Value = (u8, u32, [u16; 3]);
    let layout = Value::layout().unwrap();
    assert_eq!((layout.size, layout.alignment), (16, 4));
    let value: Value = (7, 0x0403_0201, [0x0605, 0x0807, 0x0a09]);
    let mut bytes = [0xff; 16];
    value.encode(&mut bytes);
    assert_eq!(bytes, [7, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0]);
    assert_eq!(Value::decode(&bytes), value);
    bytes[1..4].fill(0xaa);
    bytes[14..16].fill(0xbb);
    assert_eq!(Value::decode(&bytes), value);
    assert_eq!(Value::zeroed(), (0, 0, [0; 3]));
}

#[test]
fn nested_arrays_have_a_fixed_stride() {
    type Value = [(u32, u8); 2];
    let layout = Value::layout().unwrap();
    assert_eq!((layout.size, layout.alignment), (16, 4));
    let mut bytes = [0xff; 16];
    let value: Value = [(0x0403_0201, 5), (0x0908_0706, 10)];
    value.encode(&mut bytes);
    assert_eq!(bytes, [1, 2, 3, 4, 5, 0, 0, 0, 6, 7, 8, 9, 10, 0, 0, 0]);
    assert_eq!(Value::decode(&bytes), value);
    assert!(<[u32; 0]>::layout().is_err());
}

#[test]
fn scalar_encoding_preserves_all_bits() {
    let mut bytes = [0; 4];
    (-2_i32).encode(&mut bytes);
    assert_eq!(bytes, [254, 255, 255, 255]);
    assert_eq!(i32::decode(&bytes), -2);
    for bits in [0x8000_0000, 0x7f80_0000, 0x7fc0_1234, 0xff80_0000] {
        f32::from_bits(bits).encode(&mut bytes);
        assert_eq!(f32::decode(&bytes).to_bits(), bits);
    }
    assert_eq!(u16::layout().unwrap(), Layout::scalar(Scalar::U16));
}
