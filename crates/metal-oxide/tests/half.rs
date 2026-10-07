use metal_oxide::{F16, GpuValue};

#[test]
fn half_codec_preserves_all_bit_patterns_in_little_endian_order() {
    assert_eq!((F16::SIZE, F16::ALIGNMENT), (2, 2));
    let layout = F16::layout().unwrap();
    assert_eq!((layout.size, layout.alignment), (2, 2));
    assert_eq!(F16::zeroed().to_bits(), 0);
    for bits in 0..=u16::MAX {
        let value = F16::from_bits(bits);
        let mut bytes = [0xff; 2];
        value.encode(&mut bytes);
        assert_eq!(bytes, bits.to_le_bytes());
        assert_eq!(F16::decode(&bytes).to_bits(), bits);
    }
}

#[test]
fn half_aggregate_codecs_keep_two_byte_stride_and_padding() {
    let value = (7_u8, [F16::from_bits(0x3c00), F16::from_bits(0x8001)]);
    let mut bytes = [0xff; 6];
    value.encode(&mut bytes);
    assert_eq!(bytes, [7, 0, 0, 60, 1, 128]);
    bytes[1] = 0xaa;
    let decoded = <(u8, [F16; 2])>::decode(&bytes);
    assert_eq!(decoded.0, 7);
    assert_eq!(decoded.1.map(F16::to_bits), [0x3c00, 0x8001]);
}
