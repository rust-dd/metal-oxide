use super::model::{Op, Step};
use std::fmt::Write;

pub struct Word {
    pub float: bool,
    pub operation: String,
}

pub struct Printer {
    pub text: String,
    pub words: Vec<Word>,
    pub scalar: &'static str,
    pub lanes: usize,
}

impl Printer {
    pub fn program(scalar: &'static str, lanes: usize, steps: &[Step]) -> Self {
        let mut printer = Self {
            text: String::new(),
            words: Vec::new(),
            scalar,
            lanes,
        };
        for index in 0..2 {
            let values = (0..lanes)
                .map(|lane| {
                    let word = (lane + index) % 4;
                    if scalar == "f32" {
                        format!("f32::from_bits(words[{word}])")
                    } else {
                        format!("words[{word}] as {scalar}")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(
                printer.text,
                "    let v{index} = Simd::<{scalar}, {lanes}>::from_array([{values}]);"
            )
            .unwrap();
            printer.vector(&format!("v{index}"), scalar, "input");
        }
        for (index, step) in steps.iter().enumerate() {
            printer.step(index + 2, step);
        }
        printer
    }

    fn word(&mut self, expression: &str, scalar: &str, operation: &str) {
        let index = self.words.len();
        let value = if scalar == "f32" {
            format!("({expression}).to_bits()")
        } else {
            format!("({expression}) as u32")
        };
        writeln!(self.text, "    record!({index}, {value});").unwrap();
        self.words.push(Word {
            float: scalar == "f32",
            operation: operation.into(),
        });
    }

    fn vector(&mut self, variable: &str, scalar: &str, operation: &str) {
        for lane in 0..self.lanes {
            self.word(&format!("{variable}.to_array()[{lane}]"), scalar, operation);
        }
    }

    fn mask(&mut self, index: usize, expression: &str, operation: &str) {
        writeln!(self.text, "    let m{index} = {expression};").unwrap();
        self.word(&format!("m{index}.to_bitmask()"), "u64", operation);
        for lane in 0..self.lanes {
            self.word(&format!("m{index}.to_array()[{lane}]"), "bool", operation);
        }
        self.word(&format!("m{index}.any()"), "bool", operation);
        self.word(&format!("m{index}.all()"), "bool", operation);
    }

    fn step(&mut self, index: usize, step: &Step) {
        let Step {
            op,
            left,
            right,
            lane,
            indices,
        } = step;
        let ty = self.scalar;
        let lanes = self.lanes;
        let vector = format!("Simd::<{ty}, {lanes}>");
        let zero = format!("{vector}::splat(0 as {ty})");
        let operation = format!("{index}:{op:?}");
        // Finite quarter-step inputs have no varying mantissa bits below bit 17.
        let input_word = |word| {
            if ty == "f32" {
                format!("(words[{word}] >> 17)")
            } else {
                format!("words[{word}]")
            }
        };
        let expression = match op {
            Op::Array => {
                let elements = indices
                    .iter()
                    .map(|lane| format!("v{left}.to_array()[{lane}]"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{vector}::from_array([{elements}])")
            }
            Op::Splat => format!(
                "{vector}::splat(v{left}[{} as usize % {lanes}])",
                input_word(0)
            ),
            Op::ArrayReference => format!("{vector}::splat(v{left}.as_array()[{lane}])"),
            Op::Insert => {
                writeln!(self.text, "    let mut v{index} = v{left};\n    v{index}.as_mut_array()[{lane}] = v{right}[{} as usize % {lanes}];", input_word(1)).unwrap();
                String::new()
            }
            Op::Swizzle(two) => {
                let indices = indices
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                let second = if *two {
                    format!(", v{right}")
                } else {
                    String::new()
                };
                format!("simd_swizzle!(v{left}{second}, [{indices}])")
            }
            Op::Reverse => format!("v{left}.reverse()"),
            Op::Rotate(right) => format!(
                "v{left}.rotate_elements_{}::<{lane}>()",
                if *right { "right" } else { "left" }
            ),
            Op::Binary(symbol) => {
                let right = match *symbol {
                    "/" | "%" if ty == "f32" => format!("(v{right}.abs() + {vector}::splat(1.0))"),
                    "/" | "%" => format!(
                        "((v{right} & {vector}::splat(15 as {ty})) + {vector}::splat(1 as {ty}))"
                    ),
                    "<<" | ">>" => {
                        format!("(v{right} & {vector}::splat((<{ty}>::BITS - 1) as {ty}))")
                    }
                    _ => format!("v{right}"),
                };
                format!("v{left} {symbol} {right}")
            }
            Op::Method(method) => match *method {
                "neg" => format!("-v{left}"),
                "not" => format!("!v{left}"),
                "simd_min" | "simd_max" | "saturating_add" | "saturating_sub" => {
                    format!("v{left}.{method}(v{right})")
                }
                "simd_clamp" => {
                    let low = if ty.starts_with('u') { 0 } else { -17 };
                    format!(
                        "v{left}.simd_clamp({vector}::splat({low} as {ty}), {vector}::splat(19 as {ty}))"
                    )
                }
                _ => format!("v{left}.{method}().cast::<{ty}>()"),
            },
            Op::Compare(method) => {
                self.mask(index, &format!("v{left}.{method}(v{right})"), &operation);
                format!("m{index}.select(v{left}, v{right})")
            }
            Op::Mask(method) => {
                let mask_ty = match ty {
                    "i8" | "u8" => "i8",
                    "i16" | "u16" => "i16",
                    _ => "i32",
                };
                let mask = format!("Mask::<{mask_ty}, {lanes}>");
                let expression = match *method {
                    "array" => {
                        let values = (0..lanes)
                            .map(|lane| format!("{} & 1 != 0", input_word(lane)))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{mask}::from_array([{values}])")
                    }
                    "splat" => format!("{mask}::splat({} & 1 != 0)", input_word(0)),
                    "bitmask" => format!("{mask}::from_bitmask({} as u64)", input_word(0)),
                    "and" | "or" | "xor" => {
                        let symbol = match *method {
                            "and" => "&",
                            "or" => "|",
                            _ => "^",
                        };
                        format!("v{left}.simd_lt(v{right}) {symbol} v{right}.simd_ne({zero})")
                    }
                    "not" => format!("!v{left}.simd_eq(v{right})"),
                    _ => format!("v{left}.simd_le(v{right})"),
                };
                self.mask(index, &expression, &operation);
                if matches!(*method, "any" | "all") {
                    format!("{mask}::splat(m{index}.{method}()).select(v{left}, v{right})")
                } else {
                    format!("m{index}.select(v{left}, v{right})")
                }
            }
            Op::Reduction(method) => {
                let input = if ty == "f32" && matches!(*method, "reduce_sum" | "reduce_product") {
                    // Integer-valued lanes make every legal float reduction association exact.
                    format!(
                        "v{left}.simd_clamp({vector}::splat(-8.0), {vector}::splat(8.0)).cast::<i32>().cast::<f32>()"
                    )
                } else {
                    format!("v{left}")
                };
                writeln!(self.text, "    let r{index} = ({input}).{method}();").unwrap();
                self.word(&format!("r{index}"), ty, &operation);
                format!("{vector}::splat(r{index})")
            }
            Op::Cast(target) => {
                writeln!(self.text, "    let c{index} = v{left}.cast::<{target}>();").unwrap();
                self.vector(&format!("c{index}"), target, &operation);
                format!("c{index}.cast::<{ty}>()")
            }
        };
        if !expression.is_empty() {
            writeln!(self.text, "    let v{index} = {expression};").unwrap();
        }
        self.vector(&format!("v{index}"), ty, &operation);
    }
}
