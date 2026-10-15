use metal_oxide_ir::*;
use std::fmt::Write;

pub(crate) fn helpers(module: &Module) -> Result<String, Error> {
    let mut output = String::new();
    for scalar in [
        Scalar::U32,
        Scalar::I32,
        Scalar::U8,
        Scalar::U16,
        Scalar::I8,
        Scalar::I16,
        Scalar::U64,
        Scalar::Usize,
        Scalar::Isize,
    ] {
        let name = scalar.name();
        let ty = crate::expressions::scalar_name(scalar);
        if module
            .functions
            .iter()
            .any(|f| f.locals.contains(&Type::Checked(scalar)))
            || module.types.iter().any(|(_, aggregate)| {
                aggregate
                    .component_types()
                    .any(|ty| ty == Type::Checked(scalar))
            })
        {
            writeln!(
                output,
                "struct metal_oxide_checked_{name} {{ {ty} value; bool overflow; }};"
            )
            .unwrap();
        }
        for (operation, label, symbol) in [
            (BinaryOp::AddWithOverflow, "add", "+"),
            (BinaryOp::SubWithOverflow, "sub", "-"),
            (BinaryOp::MulWithOverflow, "mul", "*"),
        ] {
            let mut used = false;
            for f in &module.functions {
                for s in f.blocks.iter().flat_map(|b| &b.statements) {
                    if let Expression::Binary(op, a, _) = &s.value
                        && *op == operation
                        && operand_type(module, f, a, &s.source)? == Type::Scalar(scalar)
                    {
                        used = true;
                    }
                }
            }
            if !used {
                continue;
            }
            writeln!(output, "static inline metal_oxide_checked_{name} metal_oxide_{label}_checked_{name}({ty} a, {ty} b) {{").unwrap();
            if matches!(scalar, Scalar::I8 | Scalar::I16) {
                let minimum = -(1_i32 << (scalar.bits() - 1));
                let maximum = -minimum - 1;
                let unsigned = if scalar == Scalar::I8 {
                    "uchar"
                } else {
                    "ushort"
                };
                writeln!(output, "    int wide = int(a) {symbol} int(b);\n    return {{as_type<{ty}>({unsigned}(uint(wide))), wide < {minimum} || wide > {maximum}}};\n}}\n").unwrap();
                continue;
            }
            if matches!(scalar, Scalar::U8 | Scalar::U16) {
                let maximum = (1_u32 << scalar.bits()) - 1;
                let overflow = match operation {
                    BinaryOp::SubWithOverflow => "a < b".into(),
                    // Metal's GPU compiler can misjudge the narrow overflow intrinsic that a
                    // range test of the wide product becomes; `mulhi` stays a plain multiply.
                    BinaryOp::MulWithOverflow => format!("mulhi(a, b) != {ty}(0)"),
                    _ => format!("(uint(a) {symbol} uint(b)) > {maximum}u"),
                };
                writeln!(
                    output,
                    "    return {{{ty}(uint(a) {symbol} uint(b)), {overflow}}};\n}}\n"
                )
                .unwrap();
                continue;
            }
            if scalar == Scalar::Isize {
                output.push_str(
                    "    ulong ua = as_type<ulong>(a);\n    ulong ub = as_type<ulong>(b);\n",
                );
                writeln!(output, "    ulong value = ua {symbol} ub;").unwrap();
                let overflow = match operation {
                    BinaryOp::AddWithOverflow => "((~(ua ^ ub) & (ua ^ value)) >> 63) != 0ul",
                    BinaryOp::SubWithOverflow => "(((ua ^ ub) & (ua ^ value)) >> 63) != 0ul",
                    _ => {
                        output.push_str("    ulong ma = a < 0 ? 0ul - ua : ua;\n    ulong mb = b < 0 ? 0ul - ub : ub;\n    ulong limit = (a < 0) != (b < 0) ? 0x8000000000000000ul : 0x7ffffffffffffffful;\n");
                        "mb != 0ul && ma > limit / mb"
                    }
                };
                writeln!(
                    output,
                    "    return {{as_type<long>(value), {overflow}}};\n}}\n"
                )
                .unwrap();
                continue;
            }
            let (value, overflow) = if matches!(scalar, Scalar::U32 | Scalar::U64 | Scalar::Usize) {
                writeln!(output, "    {ty} value = a {symbol} b;").unwrap();
                let overflow = match operation {
                    BinaryOp::AddWithOverflow => "value < a",
                    BinaryOp::SubWithOverflow => "a < b",
                    BinaryOp::MulWithOverflow if scalar.bits() == 64 => "mulhi(a, b) != 0ul",
                    _ => "mulhi(a, b) != 0u",
                };
                ("value", overflow)
            } else {
                output
                    .push_str("    uint ua = as_type<uint>(a);\n    uint ub = as_type<uint>(b);\n");
                writeln!(output, "    uint value = ua {symbol} ub;").unwrap();
                let overflow = match operation {
                    BinaryOp::AddWithOverflow => {
                        "((~(ua ^ ub) & (ua ^ value)) & 0x80000000u) != 0u"
                    }
                    BinaryOp::SubWithOverflow => "(((ua ^ ub) & (ua ^ value)) & 0x80000000u) != 0u",
                    _ => {
                        output.push_str("    uint ma = a < 0 ? 0u - ua : ua;\n    uint mb = b < 0 ? 0u - ub : ub;\n    uint limit = (a < 0) != (b < 0) ? 0x80000000u : 0x7fffffffu;\n");
                        "mb != 0u && ma > limit / mb"
                    }
                };
                ("as_type<int>(value)", overflow)
            };
            writeln!(output, "    return {{{value}, {overflow}}};\n}}\n").unwrap();
        }
    }
    let mut wide = false;
    for function in &module.functions {
        for statement in function.blocks.iter().flat_map(|block| &block.statements) {
            if let Expression::Cast(value, to) = &statement.value {
                let from = operand_type(module, function, value, &statement.source)?;
                wide |= crate::arithmetic::wide_to_float(from, *to).is_some();
            }
        }
    }
    if wide {
        output.push_str(crate::arithmetic::WIDE_TO_FLOAT);
    }
    Ok(output)
}
