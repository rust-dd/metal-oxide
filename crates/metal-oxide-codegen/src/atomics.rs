use metal_oxide_ir::*;
use std::fmt::Write;

fn flags(order: AtomicOrdering, failure: AtomicOrdering) -> &'static str {
    if order == AtomicOrdering::Relaxed && failure == AtomicOrdering::Relaxed {
        ""
    } else {
        ", mem_flags::mem_device | mem_flags::mem_threadgroup"
    }
}

fn compare_name(order: AtomicOrdering, failure: AtomicOrdering) -> String {
    format!(
        "metal_oxide_compare_exchange_weak_{}_{}",
        order.name(),
        failure.name()
    )
}

pub(crate) fn expression(
    op: AtomicOp,
    order: AtomicOrdering,
    failure: AtomicOrdering,
    args: Vec<String>,
) -> String {
    let object = format!("&{}[{}]", args[0], args[1]);
    let suffix = format!("memory_order_{}{}", order.name(), flags(order, failure));
    match op {
        AtomicOp::Load => format!("atomic_load_explicit({object}, {suffix})"),
        AtomicOp::Store => format!("atomic_store_explicit({object}, {}, {suffix})", args[2]),
        AtomicOp::CompareExchangeWeak => format!(
            "{}({object}, {}, {})",
            compare_name(order, failure),
            args[2],
            args[3]
        ),
        _ => {
            let name = if matches!(
                op,
                AtomicOp::Exchange | AtomicOp::MinStore | AtomicOp::MaxStore
            ) {
                op.name().into()
            } else {
                format!("fetch_{}", op.name())
            };
            format!("atomic_{name}_explicit({object}, {}, {suffix})", args[2])
        }
    }
}

pub(crate) fn helpers(module: &Module) -> Result<String, Error> {
    let mut signatures = Vec::new();
    for function in &module.functions {
        for statement in function.blocks.iter().flat_map(|block| &block.statements) {
            if let Expression::Atomic {
                op: AtomicOp::CompareExchangeWeak,
                order,
                failure,
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
                let signature = (element, address_space, *order, *failure);
                if !signatures.contains(&signature) {
                    signatures.push(signature);
                }
            }
        }
    }
    let mut output = String::new();
    for (element, space, order, failure) in signatures {
        let ty = crate::expressions::scalar_name(element);
        let space = crate::expressions::space_name(space);
        let name = compare_name(order, failure);
        writeln!(output, "static inline metal_oxide_checked_{} {name}({space} atomic_{ty} *object, {ty} expected, {ty} desired) {{", element.name()).unwrap();
        let orders = format!(
            "memory_order_{}, memory_order_{}{}",
            order.name(),
            failure.name(),
            flags(order, failure)
        );
        if element == Scalar::F32 {
            // Metal's float CAS compares +0 and -0 as equal; integer CAS preserves Rust's bit comparison.
            writeln!(output, "    uint bits = as_type<uint>(expected);\n    bool success = atomic_compare_exchange_weak_explicit(reinterpret_cast<{space} atomic_uint *>(object), &bits, as_type<uint>(desired), {orders});\n    return {{as_type<float>(bits), success}};").unwrap();
        } else {
            writeln!(output, "    bool success = atomic_compare_exchange_weak_explicit(object, &expected, desired, {orders});\n    return {{expected, success}};").unwrap();
        }
        writeln!(output, "}}\n").unwrap();
    }
    Ok(output)
}
