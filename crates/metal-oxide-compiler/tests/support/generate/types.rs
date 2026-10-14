use super::random::Random;

/// An integer type of the generated subset; 128-bit and `i64` values are outside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int {
    pub bits: u32,
    pub signed: bool,
}

pub const INTS: [Int; 7] = [
    Int::unsigned(8),
    Int::unsigned(16),
    Int::unsigned(32),
    Int::unsigned(64),
    Int::signed(8),
    Int::signed(16),
    Int::signed(32),
];

impl Int {
    const fn unsigned(bits: u32) -> Self {
        Self {
            bits,
            signed: false,
        }
    }

    const fn signed(bits: u32) -> Self {
        Self { bits, signed: true }
    }

    pub fn name(self) -> String {
        format!("{}{}", if self.signed { 'i' } else { 'u' }, self.bits)
    }

    /// A literal that favors the edges of the range.
    pub fn literal(self, random: &mut Random) -> String {
        let name = self.name();
        let max = if self.signed {
            (1_u64 << (self.bits - 1)) - 1
        } else {
            u64::MAX >> (64 - self.bits)
        };
        match random.below(8) {
            0 => format!("0_{name}"),
            1 => format!("1_{name}"),
            2 => format!("{name}::MAX"),
            3 if self.signed => format!("{name}::MIN"),
            3 => format!("{max}_{name}"),
            4 if self.signed => format!("(-1_{name})"),
            _ => {
                let value = random.next() & max;
                if self.signed && random.chance(1, 2) && value != 0 {
                    format!("(-{value}_{name})")
                } else {
                    format!("{value}_{name}")
                }
            }
        }
    }

    /// A positive mask that keeps a divisor away from zero and, when signed, from -1.
    pub fn divisor_mask(self) -> String {
        let bits = if self.signed {
            self.bits - 2
        } else {
            self.bits
        };
        format!("{}_{}", (1_u128 << bits) - 1, self.name())
    }
}

/// A type of the generated subset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Int(Int),
    /// `f32`, kept to integers and quarters so that results are exact and normal.
    Float,
    Bool,
    Array(Box<Type>, u32),
    Tuple(Vec<Type>),
    /// An index into the program's records.
    Record(usize),
    /// An index into the program's enums.
    Enum(usize),
    Option(Box<Type>),
    /// A result whose error is a scalar, which `unwrap` can format.
    Result(Box<Type>, Box<Type>),
}

impl Type {
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Int(_) | Self::Float | Self::Bool)
    }
}

pub struct Record {
    pub name: String,
    pub fields: Vec<(String, Type)>,
}

/// An enum `E{index}` whose variants `V0, V1, ...` carry the given payloads.
pub struct Enum {
    pub variants: Vec<Vec<Type>>,
}

pub fn render(ty: &Type, records: &[Record]) -> String {
    match ty {
        Type::Int(int) => int.name(),
        Type::Float => "f32".into(),
        Type::Bool => "bool".into(),
        Type::Array(element, length) => format!("[{}; {length}]", render(element, records)),
        Type::Tuple(fields) => {
            let fields = fields
                .iter()
                .map(|field| render(field, records))
                .collect::<Vec<_>>();
            format!("({},)", fields.join(", "))
        }
        Type::Record(index) => records[*index].name.clone(),
        Type::Enum(index) => format!("E{index}"),
        Type::Option(value) => format!("Option<{}>", render(value, records)),
        Type::Result(value, error) => {
            let (value, error) = (render(value, records), render(error, records));
            format!("Result<{value}, {error}>")
        }
    }
}

/// A float literal: an integer or a quarter, in parentheses when negative.
pub fn float_literal(random: &mut Random) -> String {
    let quarters = random.below(8193) as i64 - 4096;
    let value = quarters as f32 / 4.0;
    if value < 0.0 {
        format!("({value:?}_f32)")
    } else {
        format!("{value:?}_f32")
    }
}
