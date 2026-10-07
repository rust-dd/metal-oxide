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
            writeln!(output, "inline metal_oxide_checked_{name} metal_oxide_{label}_checked_{name}({ty} a, {ty} b) {{").unwrap();
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
                    _ => format!("(uint(a) {symbol} uint(b)) > {maximum}u"),
                };
                writeln!(
                    output,
                    "    return {{{ty}(uint(a) {symbol} uint(b)), {overflow}}};\n}}\n"
                )
                .unwrap();
                continue;
            }
            let (value, overflow) = if scalar == Scalar::U32 {
                writeln!(output, "    uint value = a {symbol} b;").unwrap();
                let overflow = match operation {
                    BinaryOp::AddWithOverflow => "value < a",
                    BinaryOp::SubWithOverflow => "a < b",
                    _ => "b != 0u && a > 0xffffffffu / b",
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
    Ok(output)
}
