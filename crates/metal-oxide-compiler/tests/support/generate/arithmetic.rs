use super::{
    ast::Expr,
    builder::Builder,
    scope::Scope,
    types::{INTS, Int, Type},
};

/// What an integer method takes besides its receiver.
#[derive(Clone, Copy)]
enum Operand {
    Nothing,
    /// A value of the receiver's type.
    Value,
    /// A value of the receiver's type with its low bit set: never zero, but -1 when signed.
    Divisor,
    /// A `u32` shift amount or exponent.
    Amount,
}

use Operand::{Amount, Divisor, Nothing, Value};

const CHECKED: [(&str, Operand); 9] = [
    ("checked_add", Value),
    ("checked_sub", Value),
    ("checked_mul", Value),
    ("checked_div", Value),
    ("checked_rem", Value),
    ("checked_shl", Amount),
    ("checked_shr", Amount),
    ("checked_pow", Amount),
    ("checked_neg", Nothing),
];

const OVERFLOWING: [(&str, Operand); 7] = [
    ("overflowing_add", Value),
    ("overflowing_sub", Value),
    ("overflowing_mul", Value),
    ("overflowing_shl", Amount),
    ("overflowing_shr", Amount),
    ("overflowing_pow", Amount),
    ("overflowing_neg", Nothing),
];

/// Methods that return the receiver's type: they saturate, wrap or, for `pow`, check.
const SATURATING: [(&str, Operand); 7] = [
    ("saturating_add", Value),
    ("saturating_sub", Value),
    ("saturating_mul", Value),
    ("saturating_pow", Amount),
    ("wrapping_div", Divisor),
    ("wrapping_rem", Divisor),
    ("wrapping_pow", Amount),
];

const SIGNED_CHECKED: [(&str, Operand); 1] = [("checked_abs", Nothing)];

const SIGNED_OVERFLOWING: [(&str, Operand); 1] = [("overflowing_abs", Nothing)];

const SIGNED_SATURATING: [(&str, Operand); 4] = [
    ("saturating_neg", Nothing),
    ("saturating_abs", Nothing),
    ("saturating_div", Divisor),
    ("wrapping_abs", Nothing),
];

impl Builder {
    /// `a.checked_op(b)`, an `Option<int>` that is `None` where the operation overflows.
    pub fn checked(&mut self, scope: &Scope, int: Int, depth: u32) -> Expr {
        self.method(scope, int, depth, &CHECKED, &SIGNED_CHECKED)
    }

    /// The value of a checked operation: `unwrap_or` a default, or in fallible programs
    /// `unwrap` or `expect`, which fail where it overflows.
    pub fn checked_value(&mut self, scope: &Scope, int: Int, depth: u32) -> Expr {
        let checked = Box::new(self.checked(scope, int, depth));
        match self.random.below(4) {
            0 if self.fallible => Expr::Method(checked, "unwrap", Vec::new()),
            1 if self.fallible => {
                let message = Expr::Literal("\"generated expectation\"".into());
                Expr::Method(checked, "expect", vec![message])
            }
            _ => {
                let default = self.expr(scope, &Type::Int(int), depth);
                Expr::Method(checked, "unwrap_or", vec![default])
            }
        }
    }

    /// `a.overflowing_op(b)`, a tuple of the wrapped result and whether it overflowed.
    pub fn overflowing(&mut self, scope: &Scope, int: Int, depth: u32) -> Expr {
        self.method(scope, int, depth, &OVERFLOWING, &SIGNED_OVERFLOWING)
    }

    /// Whether an operation on a random integer type overflows.
    pub fn overflowed(&mut self, scope: &Scope, depth: u32) -> Expr {
        let int = *self.random.pick(&INTS);
        Expr::Field(Box::new(self.overflowing(scope, int, depth)), "1".into())
    }

    /// A saturating or wrapping operation, or in fallible programs `pow`, which fails where it
    /// overflows with overflow checks on.
    pub fn saturating(&mut self, scope: &Scope, int: Int, depth: u32) -> Expr {
        if self.fallible && self.random.chance(1, 6) {
            let exponent = self.operand(scope, int, depth, Amount);
            let base = Box::new(self.expr(scope, &Type::Int(int), depth));
            return Expr::Method(base, "pow", exponent.into_iter().collect());
        }
        self.method(scope, int, depth, &SATURATING, &SIGNED_SATURATING)
    }

    /// A method of `methods`, or for signed integers also of `signed`, on a value of `int`.
    fn method(
        &mut self,
        scope: &Scope,
        int: Int,
        depth: u32,
        methods: &[(&'static str, Operand)],
        signed: &[(&'static str, Operand)],
    ) -> Expr {
        let available = if int.signed {
            [methods, signed].concat()
        } else {
            methods.to_vec()
        };
        let (method, operand) = *self.random.pick(&available);
        let receiver = Box::new(self.expr(scope, &Type::Int(int), depth));
        let argument = self.operand(scope, int, depth, operand);
        Expr::Method(receiver, method, argument.into_iter().collect())
    }

    fn operand(&mut self, scope: &Scope, int: Int, depth: u32, operand: Operand) -> Option<Expr> {
        let value = |this: &mut Self| this.expr(scope, &Type::Int(int), depth);
        match operand {
            Nothing => None,
            Value => Some(value(self)),
            Divisor => {
                let one = Box::new(Expr::Literal(format!("1_{}", int.name())));
                Some(Expr::Binary(Box::new(value(self)), "|", one))
            }
            Amount => {
                let int = *self.random.pick(&INTS);
                let amount = self.expr(scope, &Type::Int(int), depth);
                Some(Expr::Cast(Box::new(amount), Type::Int(Int::U32)))
            }
        }
    }
}
