use crate::{BF16, intrinsics};
use core::cmp::Ordering;

macro_rules! binary {
    ($trait:ident, $method:ident, $bridge:ident) => {
        impl core::ops::$trait for BF16 {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                intrinsics::$bridge(self, rhs)
            }
        }
    };
}

binary!(Add, add, __metal_bf16_add);
binary!(Sub, sub, __metal_bf16_sub);
binary!(Mul, mul, __metal_bf16_mul);
binary!(Div, div, __metal_bf16_div);
binary!(Rem, rem, __metal_bf16_rem);

impl core::ops::Neg for BF16 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::from_bits(self.to_bits() ^ 0x8000)
    }
}

impl PartialEq for BF16 {
    fn eq(&self, rhs: &Self) -> bool {
        let lhs = self.to_bits();
        let rhs = rhs.to_bits();
        lhs & 0x7fff <= 0x7f80 && (lhs == rhs || (lhs | rhs) & 0x7fff == 0)
    }
}

impl PartialOrd for BF16 {
    fn lt(&self, rhs: &Self) -> bool {
        self.comparable(*rhs) && self.ordered_bits() < rhs.ordered_bits()
    }
    fn le(&self, rhs: &Self) -> bool {
        self.comparable(*rhs) && self.ordered_bits() <= rhs.ordered_bits()
    }
    fn gt(&self, rhs: &Self) -> bool {
        self.comparable(*rhs) && self.ordered_bits() > rhs.ordered_bits()
    }
    fn ge(&self, rhs: &Self) -> bool {
        self.comparable(*rhs) && self.ordered_bits() >= rhs.ordered_bits()
    }
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        if !self.comparable(*rhs) {
            return None;
        }
        let lhs = self.ordered_bits();
        let rhs = rhs.ordered_bits();
        Some(if lhs < rhs {
            Ordering::Less
        } else if lhs > rhs {
            Ordering::Greater
        } else {
            Ordering::Equal
        })
    }
}

impl BF16 {
    fn comparable(self, rhs: Self) -> bool {
        self.to_bits() & 0x7fff <= 0x7f80 && rhs.to_bits() & 0x7fff <= 0x7f80
    }

    fn ordered_bits(self) -> i16 {
        // Integer ordering preserves subnormals even when Metal flushes f32 comparisons.
        let bits = self.to_bits() as i16;
        if bits < 0 {
            i16::MIN.wrapping_sub(bits)
        } else {
            bits
        }
    }
}

macro_rules! assign {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl core::ops::$trait for BF16 {
            fn $method(&mut self, rhs: Self) { *self = *self $operator rhs; }
        }
    };
}

assign!(AddAssign, add_assign, +);
assign!(SubAssign, sub_assign, -);
assign!(MulAssign, mul_assign, *);
assign!(DivAssign, div_assign, /);
assign!(RemAssign, rem_assign, %);
