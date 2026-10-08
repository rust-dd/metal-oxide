use metal_oxide_ir::*;
use std::fmt::Write;

pub(crate) fn expression(op: AtomicOp, args: Vec<String>) -> String {
    let object = format!("&{}[{}]", args[0], args[1]);
    match op {
        AtomicOp::Load => format!("atomic_load_explicit({object}, memory_order_relaxed)"),
        AtomicOp::Store => format!(
            "atomic_store_explicit({object}, {}, memory_order_relaxed)",
            args[2]
        ),
        AtomicOp::CompareExchangeWeak => format!(
            "metal_oxide_compare_exchange_weak({object}, {}, {})",
            args[2], args[3]
        ),
        _ => {
            let name = if op == AtomicOp::Exchange {
                "exchange".into()
            } else {
                format!("fetch_{}", op.name())
            };
            format!(
                "atomic_{name}_explicit({object}, {}, memory_order_relaxed)",
                args[2]
            )
        }
    }
}

pub(crate) fn helpers(module: &Module) -> Result<String, Error> {
    let mut signatures = Vec::new();
    for function in &module.functions {
        for statement in function.blocks.iter().flat_map(|block| &block.statements) {
            if let Expression::Atomic {
                op: AtomicOp::CompareExchangeWeak,
                arguments,
            } = &statement.value
            {
                let Type::Buffer {
                    element: Element::Scalar(element),
                    address_space,
                    ..
                } = operand_type(module, function, &arguments[0], &statement.source)?
                else {
                    unreachable!()
                };
                if !signatures.contains(&(element, address_space)) {
                    signatures.push((element, address_space));
                }
            }
        }
    }
    let mut output = String::new();
    for (element, space) in signatures {
        let ty = crate::expressions::scalar_name(element);
        let space = match space {
            AddressSpace::Device => "device",
            AddressSpace::Threadgroup => "threadgroup",
        };
        writeln!(output, "inline metal_oxide_checked_{} metal_oxide_compare_exchange_weak({space} atomic_{ty} *object, {ty} expected, {ty} desired) {{\n    bool success = atomic_compare_exchange_weak_explicit(object, &expected, desired, memory_order_relaxed, memory_order_relaxed);\n    return {{expected, success}};\n}}\n", element.name()).unwrap();
    }
    Ok(output)
}
