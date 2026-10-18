use metal_oxide_ir::*;

use crate::{expressions::scalar_name, scan::uses};

/// MSL for a matrix operation over operands of these types.
pub(crate) fn expression(op: MatrixOp, types: &[Type], arguments: &[String]) -> String {
    match op {
        MatrixOp::Splat => format!(
            "make_filled_simdgroup_matrix<{}, 8, 8>({})",
            element_name(types[0]),
            arguments[0]
        ),
        MatrixOp::MultiplyAccumulate => format!(
            "metal_oxide_matrix_multiply_accumulate({})",
            arguments.join(", ")
        ),
        // Metal takes the origin as (column, row).
        MatrixOp::Load { transposed } => format!(
            "metal_oxide_matrix_load<{}>({}, ulong({}), ulong2({}, {}), {transposed})",
            element_name(types[0]),
            arguments[0],
            arguments[3],
            arguments[2],
            arguments[1]
        ),
        MatrixOp::Store { transposed } => format!(
            "simdgroup_store({}, {}, ulong({}), ulong2({}, {}), {transposed})",
            arguments[4], arguments[0], arguments[3], arguments[2], arguments[1]
        ),
    }
}

/// The MSL element type of a splat value or of the buffer a matrix is loaded from.
fn element_name(ty: Type) -> &'static str {
    match ty {
        Type::Scalar(scalar)
        | Type::Buffer {
            element: Element::Scalar(scalar),
            ..
        } => scalar_name(scalar),
        _ => unreachable!("validated matrix operand"),
    }
}

/// Value-returning forms of the Metal matrix operations that write an out-parameter.
pub(crate) fn helpers(module: &Module) -> &'static str {
    let used = uses(module, |value| {
        matches!(
            value,
            Expression::Matrix {
                op: MatrixOp::Load { .. } | MatrixOp::MultiplyAccumulate,
                ..
            }
        )
    });
    if used { HELPERS } else { "" }
}

const HELPERS: &str = "template <typename T, typename P>
static inline simdgroup_matrix<T, 8, 8> metal_oxide_matrix_load(P source, ulong stride, ulong2 origin, bool transpose) {
    simdgroup_matrix<T, 8, 8> matrix;
    simdgroup_load(matrix, source, stride, origin, transpose);
    return matrix;
}

template <typename R, typename A, typename B>
static inline simdgroup_matrix<R, 8, 8> metal_oxide_matrix_multiply_accumulate(simdgroup_matrix<A, 8, 8> a, simdgroup_matrix<B, 8, 8> b, simdgroup_matrix<R, 8, 8> c) {
    simdgroup_matrix<R, 8, 8> result;
    simdgroup_multiply_accumulate(result, a, b, c);
    return result;
}

";
