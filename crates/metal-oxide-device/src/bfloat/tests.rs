use super::BF16;

#[test]
fn bit_conversions_preserve_all_representations_and_support_const_evaluation() {
    const NEGATIVE_ZERO: BF16 = BF16::from_bits(0x8000);
    const BITS: u16 = NEGATIVE_ZERO.to_bits();
    assert_eq!(BITS, 0x8000);
    for bits in 0..=u16::MAX {
        assert_eq!(BF16::from_bits(bits).to_bits(), bits);
    }
}

#[test]
fn comparisons_match_reference_for_every_encoding_and_ordering_boundary() {
    let boundaries = [
        0x0000, 0x0001, 0x0002, 0x003f, 0x0040, 0x007e, 0x007f, 0x0080, 0x0081, 0x3f7f, 0x3f80,
        0x3f81, 0x7f7e, 0x7f7f, 0x7f80, 0x7f81, 0x7fbf, 0x7fc0, 0x7fff,
    ];
    for lhs in 0..=u16::MAX {
        for rhs in [lhs, lhs.wrapping_add(1), lhs.wrapping_sub(1), lhs ^ 0x8000] {
            compare(lhs, rhs);
        }
        for rhs in boundaries {
            compare(lhs, rhs);
            compare(lhs, rhs | 0x8000);
        }
    }
}

fn compare(lhs: u16, rhs: u16) {
    let a = BF16::from_bits(lhs);
    let b = BF16::from_bits(rhs);
    let x = half::bf16::from_bits(lhs);
    let y = half::bf16::from_bits(rhs);
    assert_eq!(
        (
            a == b,
            a != b,
            a < b,
            a <= b,
            a > b,
            a >= b,
            a.partial_cmp(&b)
        ),
        (
            x == y,
            x != y,
            x < y,
            x <= y,
            x > y,
            x >= y,
            x.partial_cmp(&y)
        ),
        "BF16 comparison {lhs:#06x} with {rhs:#06x}"
    );
}

#[test]
fn abs_clears_only_the_sign_bit_of_every_value() {
    const ABS: u16 = BF16::from_bits(0x8000).abs().to_bits();
    assert_eq!(ABS, 0);
    for bits in 0..=u16::MAX {
        assert_eq!(BF16::from_bits(bits).abs().to_bits(), bits & 0x7fff);
    }
}
