use super::{
    ast::Expr,
    builder::{Builder, Scope},
    types::{INTS, Int, Type, float_literal},
};

const COMPARISONS: [&str; 6] = ["==", "!=", "<", "<=", ">", ">="];

impl Builder {
    /// An expression of `ty`, at most `depth` operators deep.
    pub fn expr(&mut self, scope: &Scope, ty: &Type, depth: u32) -> Expr {
        if depth == 0 || self.random.chance(1, 4) {
            return self.leaf(scope, ty);
        }
        if self.random.chance(1, 8)
            && let Some(call) = self.call(scope, ty, depth)
        {
            return call;
        }
        match ty {
            Type::Int(int) => self.int(scope, *int, depth - 1),
            Type::Float => self.float(scope, depth - 1),
            Type::Bool => self.boolean(scope, depth - 1),
            _ if self.random.chance(1, 3) => {
                let condition = self.expr(scope, &Type::Bool, depth - 1);
                let then = self.expr(scope, ty, depth - 1);
                let otherwise = self.expr(scope, ty, depth - 1);
                Expr::If(Box::new(condition), Box::new(then), Box::new(otherwise))
            }
            _ => self.construct(scope, ty, depth - 1),
        }
    }

    /// A literal, a local or a projection of one, or for `u32` the helper's const parameter.
    fn leaf(&mut self, scope: &Scope, ty: &Type) -> Expr {
        let mut found = scope.places(&self.records, ty);
        if *ty
            == Type::Int(Int {
                bits: 32,
                signed: false,
            })
            && let Some(generic) = &scope.generic
        {
            found.push(Expr::Local(generic.clone()));
        }
        if !found.is_empty() && self.random.chance(3, 4) {
            let index = self.random.below(found.len() as u64) as usize;
            return self.indexed(scope, found.swap_remove(index));
        }
        match ty {
            Type::Int(int) => Expr::Literal(int.literal(&mut self.random)),
            Type::Float => Expr::Literal(float_literal(&mut self.random)),
            Type::Bool => Expr::Literal(self.random.chance(1, 2).to_string()),
            _ => self.construct(scope, ty, 0),
        }
    }

    /// Fills the index operands that [`Scope::places`] leaves empty.
    fn indexed(&mut self, scope: &Scope, expr: Expr) -> Expr {
        match expr {
            Expr::Field(value, field) => Expr::Field(Box::new(self.indexed(scope, *value)), field),
            Expr::Index(value, _, modulus) => {
                let int = *self.random.pick(&INTS);
                let (index, modulus) = match modulus {
                    Some(length) if self.fallible && self.random.chance(1, 4) => {
                        let index = self.varying(scope, int);
                        (index, (!self.random.chance(1, 4)).then_some(length + 1))
                    }
                    modulus => (self.leaf(scope, &Type::Int(int)), modulus),
                };
                Expr::Index(
                    Box::new(self.indexed(scope, *value)),
                    Box::new(index),
                    modulus,
                )
            }
            other => other,
        }
    }

    fn construct(&mut self, scope: &Scope, ty: &Type, depth: u32) -> Expr {
        let fields = match ty {
            Type::Array(element, length) => vec![(**element).clone(); *length as usize],
            Type::Tuple(fields) => fields.clone(),
            Type::Record(index) => self.records[*index]
                .fields
                .iter()
                .map(|(_, ty)| ty.clone())
                .collect(),
            _ => unreachable!("scalars have literals"),
        };
        let values = fields
            .iter()
            .map(|field| self.expr(scope, field, depth))
            .collect();
        Expr::Aggregate(ty.clone(), values)
    }

    fn call(&mut self, scope: &Scope, ty: &Type, depth: u32) -> Option<Expr> {
        let callees = self
            .helpers
            .iter()
            .enumerate()
            .filter(|(_, helper)| helper.result == *ty)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if callees.is_empty() {
            return None;
        }
        let index = *self.random.pick(&callees);
        let (name, generic, parameters) = {
            let helper = &self.helpers[index];
            (
                helper.name.clone(),
                helper.generic.is_some(),
                helper.parameters.clone(),
            )
        };
        let generic = generic.then(|| self.random.below(64) as u32);
        let arguments = parameters
            .iter()
            .map(|(_, ty)| self.expr(scope, ty, depth - 1))
            .collect();
        Some(Expr::Call(name, generic, arguments))
    }

    fn int(&mut self, scope: &Scope, int: Int, depth: u32) -> Expr {
        let ty = Type::Int(int);
        let operand = |this: &mut Self| Box::new(this.expr(scope, &ty, depth));
        if self.fallible && self.random.chance(1, 10) {
            let varying = Box::new(self.varying(scope, int));
            if int.signed && self.random.chance(1, 8) {
                return Expr::Unary("-", varying);
            }
            let operator = *self.random.pick(&["+", "-", "*", "/", "%", "<<", ">>"]);
            return if matches!(operator, "+" | "-" | "*") && self.random.chance(1, 2) {
                Expr::Binary(varying, operator, operand(self))
            } else {
                Expr::Binary(operand(self), operator, varying)
            };
        }
        match self.random.below(11) {
            0 => {
                let method = *self
                    .random
                    .pick(&["wrapping_add", "wrapping_sub", "wrapping_mul"]);
                Expr::Method(operand(self), method, vec![*operand(self)])
            }
            1 => Expr::Binary(
                operand(self),
                self.random.pick(&["^", "&", "|"]),
                operand(self),
            ),
            2 => {
                let method = *self.random.pick(&[
                    "wrapping_shl",
                    "wrapping_shr",
                    "rotate_left",
                    "rotate_right",
                ]);
                let amount = self.any_int(scope, depth);
                let amount = Expr::Cast(
                    Box::new(amount),
                    Type::Int(Int {
                        bits: 32,
                        signed: false,
                    }),
                );
                Expr::Method(operand(self), method, vec![amount])
            }
            3 => {
                let operator = *self.random.pick(&["/", "%"]);
                let divisor = if int.signed {
                    let mask = Expr::Literal(int.divisor_mask());
                    Expr::Binary(operand(self), "&", Box::new(mask))
                } else {
                    *operand(self)
                };
                let one = Expr::Literal(format!("1_{}", int.name()));
                let divisor = Expr::Binary(Box::new(divisor), "|", Box::new(one));
                Expr::Binary(operand(self), operator, Box::new(divisor))
            }
            4 => Expr::Method(
                operand(self),
                self.random.pick(&["min", "max"]),
                vec![*operand(self)],
            ),
            5 => match self.random.below(3) {
                0 => Expr::Unary("!", operand(self)),
                1 => Expr::Method(operand(self), "reverse_bits", vec![]),
                _ => Expr::Method(operand(self), "wrapping_neg", vec![]),
            },
            6 if int
                == (Int {
                    bits: 32,
                    signed: false,
                }) =>
            {
                let method = *self
                    .random
                    .pick(&["count_ones", "leading_zeros", "trailing_zeros"]);
                Expr::Method(Box::new(self.any_int(scope, depth)), method, vec![])
            }
            7 => {
                let source = match self.random.below(4) {
                    0 => Expr::Cast(
                        Box::new(self.expr(scope, &Type::Bool, depth)),
                        Type::Int(Int {
                            bits: 8,
                            signed: false,
                        }),
                    ),
                    1 => self.expr(scope, &Type::Float, depth),
                    _ => self.any_int(scope, depth),
                };
                Expr::Cast(Box::new(source), ty)
            }
            8 => {
                let condition = self.expr(scope, &Type::Bool, depth);
                Expr::If(Box::new(condition), operand(self), operand(self))
            }
            9 => {
                let scrutinee = self.any_int(scope, depth);
                let arms = [
                    *operand(self),
                    *operand(self),
                    *operand(self),
                    *operand(self),
                ];
                Expr::Match(Box::new(scrutinee), Box::new(arms))
            }
            _ => Expr::Method(operand(self), "wrapping_add", vec![*operand(self)]),
        }
    }

    /// An integer of `int` that differs between threads, from a parameter when there is one.
    pub fn varying(&mut self, scope: &Scope, int: Int) -> Expr {
        if scope.varying.is_empty() {
            return self.leaf(scope, &Type::Int(int));
        }
        let name = self.random.pick(&scope.varying).clone();
        Expr::Cast(Box::new(Expr::Local(name)), Type::Int(int))
    }

    fn any_int(&mut self, scope: &Scope, depth: u32) -> Expr {
        let int = *self.random.pick(&INTS);
        self.expr(scope, &Type::Int(int), depth)
    }

    fn float(&mut self, scope: &Scope, depth: u32) -> Expr {
        let operand = |this: &mut Self| Box::new(this.expr(scope, &Type::Float, depth));
        match self.random.below(6) {
            0 | 1 => Expr::Binary(
                operand(self),
                self.random.pick(&["+", "-", "*"]),
                operand(self),
            ),
            2 => Expr::Unary("-", operand(self)),
            3 => match self.random.below(3) {
                0 => Expr::Method(operand(self), "abs", vec![]),
                1 => Expr::Method(operand(self), "min", vec![*operand(self)]),
                _ => Expr::Method(operand(self), "max", vec![*operand(self)]),
            },
            4 => Expr::Cast(Box::new(self.any_int(scope, depth)), Type::Float),
            _ => {
                let condition = self.expr(scope, &Type::Bool, depth);
                Expr::If(Box::new(condition), operand(self), operand(self))
            }
        }
    }

    fn boolean(&mut self, scope: &Scope, depth: u32) -> Expr {
        match self.random.below(5) {
            0 | 1 => {
                let ty = if self.random.chance(1, 4) {
                    Type::Float
                } else {
                    Type::Int(*self.random.pick(&INTS))
                };
                let left = self.expr(scope, &ty, depth);
                let right = self.expr(scope, &ty, depth);
                Expr::Binary(
                    Box::new(left),
                    self.random.pick(&COMPARISONS),
                    Box::new(right),
                )
            }
            2 => Expr::Unary("!", Box::new(self.expr(scope, &Type::Bool, depth))),
            3 => {
                let left = self.expr(scope, &Type::Bool, depth);
                let right = self.expr(scope, &Type::Bool, depth);
                Expr::Binary(
                    Box::new(left),
                    self.random.pick(&["&&", "||"]),
                    Box::new(right),
                )
            }
            _ => {
                let condition = self.expr(scope, &Type::Bool, depth);
                let then = self.expr(scope, &Type::Bool, depth);
                let otherwise = self.expr(scope, &Type::Bool, depth);
                Expr::If(Box::new(condition), Box::new(then), Box::new(otherwise))
            }
        }
    }
}
