use crate::{BF16, intrinsics};

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
        self.to_f32() == rhs.to_f32()
    }
}

impl PartialOrd for BF16 {
    fn lt(&self, rhs: &Self) -> bool {
        self.to_f32() < rhs.to_f32()
    }
    fn le(&self, rhs: &Self) -> bool {
        self.to_f32() <= rhs.to_f32()
    }
    fn gt(&self, rhs: &Self) -> bool {
        self.to_f32() > rhs.to_f32()
    }
    fn ge(&self, rhs: &Self) -> bool {
        self.to_f32() >= rhs.to_f32()
    }
    fn partial_cmp(&self, rhs: &Self) -> Option<core::cmp::Ordering> {
        self.to_f32().partial_cmp(&rhs.to_f32())
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
