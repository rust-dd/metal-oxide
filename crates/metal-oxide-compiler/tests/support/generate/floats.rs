use super::{
    ast::Expr,
    builder::Builder,
    scope::Scope,
    types::{Int, Type, float_literal},
};

/// Math functions of one argument and the ULPs their results may differ by: Metal's precise
/// bound and one for the CPU's libm.
const UNARY: [(&str, u32); 12] = [
    ("exp", 5),
    ("exp2", 5),
    ("ln", 5),
    ("log2", 5),
    ("log10", 5),
    ("sin", 5),
    ("cos", 5),
    ("tan", 7),
    ("asin", 5),
    ("acos", 5),
    ("atan", 6),
    ("tanh", 6),
];

const BINARY: [(&str, u32); 3] = [("atan2", 7), ("powf", 17), ("hypot", 4)];

impl Builder {
    /// An `f32` from correctly rounded operations, which Metal computes bit for bit like Rust.
    pub fn float(&mut self, scope: &Scope, depth: u32) -> Expr {
        let operand = |this: &mut Self| Box::new(this.expr(scope, &Type::Float, depth));
        match self.random.below(8) {
            0 | 1 => {
                let operator = *self.random.pick(&["+", "-", "*", "/", "%"]);
                Expr::Binary(operand(self), operator, operand(self))
            }
            2 => Expr::Unary("-", operand(self)),
            3 => match self.random.below(3) {
                0 => Expr::Method(operand(self), "abs", vec![]),
                1 => Expr::Extreme("min", operand(self), operand(self)),
                _ => Expr::Extreme("max", operand(self), operand(self)),
            },
            4 => Expr::Cast(Box::new(self.any_int(scope, depth)), Type::Float),
            5 => {
                let methods = ["sqrt", "floor", "ceil", "round", "trunc", "signum"];
                Expr::Method(operand(self), self.random.pick(&methods), vec![])
            }
            6 => match self.random.below(3) {
                0 => Expr::Method(
                    operand(self),
                    "mul_add",
                    vec![*operand(self), *operand(self)],
                ),
                // A NaN's sign differs between the CPU and Metal, so the sign comes from a literal.
                1 => {
                    let sign = Expr::Literal(float_literal(&mut self.random));
                    Expr::Method(operand(self), "copysign", vec![sign])
                }
                _ => {
                    let mut bounds = [
                        float_literal(&mut self.random),
                        float_literal(&mut self.random),
                    ];
                    bounds.sort_by(|a, b| literal_value(a).total_cmp(&literal_value(b)));
                    let bounds = bounds.map(Expr::Literal).to_vec();
                    Expr::Method(operand(self), "clamp", bounds)
                }
            },
            _ => {
                let condition = self.expr(scope, &Type::Bool, depth);
                Expr::If(Box::new(condition), operand(self), operand(self))
            }
        }
    }

    /// A math function of correctly rounded operands, and the ULPs its result may differ by.
    pub fn approximation(&mut self, scope: &Scope) -> (Expr, u32) {
        let operand = |this: &mut Self| Box::new(this.expr(scope, &Type::Float, 3));
        match self.random.below(6) {
            // Rust leaves `powi`'s rounding open: LLVM folds constants in double precision.
            2 => {
                let exponent = self.expr(scope, &Type::Int(Int::U8), 2);
                let exponent = Expr::Binary(
                    Box::new(exponent),
                    "%",
                    Box::new(Expr::Literal("9_u8".into())),
                );
                let exponent = Expr::Binary(
                    Box::new(Expr::Cast(Box::new(exponent), Type::Int(Int::I32))),
                    "-",
                    Box::new(Expr::Literal("4_i32".into())),
                );
                (Expr::Method(operand(self), "powi", vec![exponent]), 5)
            }
            0 => {
                let (method, ulps) = *self.random.pick(&BINARY);
                (
                    Expr::Method(operand(self), method, vec![*operand(self)]),
                    ulps,
                )
            }
            1 => {
                let pair = Expr::Method(operand(self), "sin_cos", Vec::new());
                let field = self.random.pick(&["0", "1"]).to_string();
                (Expr::Field(Box::new(pair), field), 5)
            }
            _ => {
                let (method, ulps) = *self.random.pick(&UNARY);
                (Expr::Method(operand(self), method, Vec::new()), ulps)
            }
        }
    }
}

/// The value of a literal that [`float_literal`] wrote.
fn literal_value(literal: &str) -> f32 {
    let digits = literal.trim_start_matches('(').trim_end_matches(')');
    digits.trim_end_matches("_f32").parse().unwrap()
}
