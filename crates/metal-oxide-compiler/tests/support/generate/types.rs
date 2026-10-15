use super::random::Random;

/// An integer type of the generated subset; 128-bit and `i64` values are outside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int {
    pub bits: u32,
    pub signed: bool,
}

pub const U32: Type = Type::Int(Int::U32);

pub const I32: Type = Type::Int(Int::I32);

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
    pub const U32: Self = Self::unsigned(32);
    pub const I32: Self = Self::signed(32);

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
    /// `&T`, or `&mut T` when set; only helper parameters are references.
    Ref(Box<Type>, bool),
    /// `&[T]`, or `&mut [T]` when set, borrowing part of an array local.
    Slice(Box<Type>, bool),
    /// A closure local, or a helper's `impl Fn` parameter, with its parameter and result types.
    Closure(Vec<Type>, Box<Type>),
    /// Threadgroup memory of the given length that the kernel allocates.
    Threadgroup(Box<Type>, u32),
    /// Threadgroup atomics of the given length that the kernel allocates.
    AtomicThreadgroup(Int, u32),
    /// Device atomics, a kernel parameter.
    Atomic(Int),
}

#[derive(Clone)]
pub struct Record {
    pub name: String,
    pub fields: Vec<(String, Type)>,
}

/// An enum `E{index}` whose variants `V0, V1, ...` carry the given payloads.
#[derive(Clone)]
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
        Type::Ref(target, mutable) => {
            format!("&{}{}", mutability(*mutable), render(target, records))
        }
        Type::Slice(element, mutable) => {
            format!("&{}[{}]", mutability(*mutable), render(element, records))
        }
        Type::Closure(parameters, result) => {
            let parameters = parameters
                .iter()
                .map(|ty| render(ty, records))
                .collect::<Vec<_>>();
            format!(
                "impl Fn({}) -> {}",
                parameters.join(", "),
                render(result, records)
            )
        }
        Type::Threadgroup(element, length) => {
            format!("ThreadgroupBuffer<{}, {length}>", render(element, records))
        }
        Type::AtomicThreadgroup(int, length) => {
            format!("AtomicThreadgroupBuffer<{}, {length}>", int.name())
        }
        Type::Atomic(int) => format!("AtomicBuffer<{}>", int.name()),
    }
}

/// Whether values of `ty` take no bytes, as a single variant without payload does. The
/// compiler rejects slices of such elements, so iterators and slices avoid them.
pub fn zero_sized(ty: &Type, records: &[Record], enums: &[Enum]) -> bool {
    let all = |types: &[Type]| types.iter().all(|ty| zero_sized(ty, records, enums));
    match ty {
        Type::Array(element, length) => *length == 0 || zero_sized(element, records, enums),
        Type::Tuple(fields) => all(fields),
        Type::Record(index) => records[*index]
            .fields
            .iter()
            .all(|(_, ty)| zero_sized(ty, records, enums)),
        Type::Enum(index) => match enums[*index].variants.as_slice() {
            [payload] => all(payload),
            _ => false,
        },
        _ => false,
    }
}

pub fn mutability(mutable: bool) -> &'static str {
    if mutable { "mut " } else { "" }
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
