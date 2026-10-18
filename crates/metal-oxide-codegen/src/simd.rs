use metal_oxide_ir::*;

use crate::interface::uses;

pub(crate) fn expression(op: SimdOp, arguments: Vec<String>) -> String {
    let name = match op {
        SimdOp::InclusiveSum => "simd_prefix_inclusive_sum".into(),
        SimdOp::ExclusiveSum => "simd_prefix_exclusive_sum".into(),
        SimdOp::Ballot => {
            return format!(
                "metal_oxide_ballot({}, metal_oxide_ctx.simd_size)",
                arguments[0]
            );
        }
        _ => format!("simd_{}", op.name()),
    };
    format!("{name}({})", arguments.join(", "))
}

pub(crate) fn helpers(module: &Module) -> String {
    let ballot = uses(module, |value| {
        matches!(
            value,
            Expression::Simd {
                op: SimdOp::Ballot,
                ..
            }
        )
    });
    if !ballot {
        return String::new();
    }
    let ty =
        crate::expressions::type_name(ballot_type(&module.types).expect("validated ballot result"));
    format!(
        "static inline {ty} metal_oxide_ballot(bool predicate, uint width) {{\n    ulong bits = ulong(simd_ballot(predicate));\n    if (width < 64u) bits &= (1ul << width) - 1ul;\n    return {{{{uint(bits), uint(bits >> 32u)}}}};\n}}\n\n"
    )
}
