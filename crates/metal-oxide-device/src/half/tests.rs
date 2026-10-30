use super::F16;

#[test]
fn bit_conversions_preserve_all_representations_and_support_const_evaluation() {
    const NEGATIVE_ZERO: F16 = F16::from_bits(0x8000);
    const BITS: u16 = NEGATIVE_ZERO.to_bits();
    assert_eq!(BITS, 0x8000);
    for bits in 0..=u16::MAX {
        assert_eq!(F16::from_bits(bits).to_bits(), bits);
    }
}

#[test]
fn abs_clears_only_the_sign_bit_of_every_value() {
    const ABS: u16 = F16::from_bits(0x8000).abs().to_bits();
    assert_eq!(ABS, 0);
    for bits in 0..=u16::MAX {
        assert_eq!(F16::from_bits(bits).abs().to_bits(), bits & 0x7fff);
    }
}
