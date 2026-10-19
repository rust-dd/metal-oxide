#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// A tensor's extent(0) varies fastest, so a row-major matrix has extents (columns, rows).
using extents2 = dextents<int32_t, 2>;

// The header documents C = A * B + C, while the default multiply mode overwrites the product;
// a zeroed product is A * B under either, so each repeat overwrites C.
template <typename Op, typename Left, typename Right, typename Destination>
static inline void multiply(
    thread const Op &op,
    thread Left &left,
    thread Right &right,
    thread Destination &destination) {
    auto product = op.template get_destination_cooperative_tensor<Left, Right, float>();
#pragma unroll
    for (uint16_t i = 0; i < product.get_capacity(); ++i)
        if (product.is_valid_element(i))
            product[i] = 0.0f;
    op.run(left, right, product);
    product.store(destination);
}

// A block of four SIMD groups (128 threads) computes a ROWS x COLUMNS tile of C; edge blocks
// slice to the remaining extents, which matmul2d checks.
template <typename T, int ROWS, int COLUMNS>
static inline void gemm_tile(
    device const T *a,
    device const T *b,
    device float *c,
    int m,
    int n,
    int k,
    uint3 block) {
    constexpr auto descriptor = matmul2d_descriptor(
        ROWS, COLUMNS, static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    // MPP rejects const element types; the kernel only reads A and B.
    tensor<device T, extents2, tensor_inline> left(const_cast<device T *>(a), extents2(k, m));
    tensor<device T, extents2, tensor_inline> right(const_cast<device T *>(b), extents2(n, k));
    tensor<device float, extents2, tensor_inline> result(c, extents2(n, m));
    int row = block.y * ROWS;
    int column = block.x * COLUMNS;
    if (row + ROWS <= m && column + COLUMNS <= n) {
        auto tile_a = left.template slice<dynamic_extent, ROWS>(0, row);
        auto tile_b = right.template slice<COLUMNS, dynamic_extent>(column, 0);
        auto tile_c = result.template slice<COLUMNS, ROWS>(column, row);
        multiply(op, tile_a, tile_b, tile_c);
    } else {
        auto tile_a = left.slice(0, row);
        auto tile_b = right.slice(column, 0);
        auto tile_c = result.slice(column, row);
        multiply(op, tile_a, tile_b, tile_c);
    }
}

// The runner's grid (TILE in benchmarks/runner/src/gpu/gemm.rs) assumes this tile.
constant constexpr int ROWS = 64;
constant constexpr int COLUMNS = 64;

kernel void gemm_f32_mpp(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    uint3 block [[threadgroup_position_in_grid]]) {
    gemm_tile<float, ROWS, COLUMNS>(a, b, c, m, n, k, block);
}

kernel void gemm_f16_mpp(
    device const half *a [[buffer(0)]],
    device const half *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    uint3 block [[threadgroup_position_in_grid]]) {
    gemm_tile<half, ROWS, COLUMNS>(a, b, c, m, n, k, block);
}
