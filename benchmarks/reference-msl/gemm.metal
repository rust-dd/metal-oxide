#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

// Literal fragment indices keep the accumulators in registers; loops over an array of them
// ran up to 2.8x slower.
#define FRAGMENTS(X) \
    X(0, 0) X(0, 1) X(0, 2) X(0, 3) X(1, 0) X(1, 1) X(1, 2) X(1, 3) \
    X(2, 0) X(2, 1) X(2, 2) X(2, 3) X(3, 0) X(3, 1) X(3, 2) X(3, 3)
#define ZERO(i, j) acc[i][j] = zero;
#define MULTIPLY(i, j) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);
#define STORE(i, j) \
    simdgroup_store(acc[i][j], c, n, ulong2(column0 + column + 8 * j, row0 + row + 8 * i));
#define STAGE(i, j) simdgroup_store(acc[i][j], tile_c, 32, ulong2(8 * j, 8 * i));

// Twin of examples/gemm/kernels: a block computes a 64x64 tile of C, and each of its four
// SIMD groups takes a 32x32 quarter as 4x4 fragments of 8x8.
template <typename T>
static inline void gemm_block(
    device const T *a,
    device const T *b,
    device float *c,
    uint m,
    uint n,
    uint k,
    threadgroup T *tile_a,
    threadgroup T *tile_b,
    threadgroup float *tile_c,
    uint tid,
    uint group,
    uint3 block) {
    uint row0 = block.y * 64;
    uint column0 = block.x * 64;
    uint row = group / 2 * 32;
    uint column = group % 2 * 32;
    simdgroup_matrix<float, 8, 8> zero = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc[4][4];
    FRAGMENTS(ZERO)
    for (uint base = 0; base < k; base += 16) {
        for (uint step = 0; step < 8; ++step) {
            uint slot = tid + 128 * step;
            uint a_row = row0 + slot / 16;
            uint a_inner = base + slot % 16;
            tile_a[slot] = (a_row < m && a_inner < k) ? a[a_row * k + a_inner] : T(0);
            uint b_inner = base + slot / 64;
            uint b_column = column0 + slot % 64;
            tile_b[slot] = (b_inner < k && b_column < n) ? b[b_inner * n + b_column] : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint slice = 0; slice < 2; ++slice) {
            uint inner = 8 * slice;
            simdgroup_matrix<T, 8, 8> fa[4];
            simdgroup_matrix<T, 8, 8> fb[4];
            simdgroup_load(fa[0], tile_a, 16, ulong2(inner, row));
            simdgroup_load(fa[1], tile_a, 16, ulong2(inner, row + 8));
            simdgroup_load(fa[2], tile_a, 16, ulong2(inner, row + 16));
            simdgroup_load(fa[3], tile_a, 16, ulong2(inner, row + 24));
            simdgroup_load(fb[0], tile_b, 64, ulong2(column, inner));
            simdgroup_load(fb[1], tile_b, 64, ulong2(column + 8, inner));
            simdgroup_load(fb[2], tile_b, 64, ulong2(column + 16, inner));
            simdgroup_load(fb[3], tile_b, 64, ulong2(column + 24, inner));
            FRAGMENTS(MULTIPLY)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 + 64 <= m && column0 + 64 <= n) {
        FRAGMENTS(STORE)
        return;
    }
    // The groups take turns through one 32x32 tile: shader validation doubles threadgroup memory.
    for (uint turn = 0; turn < 4; ++turn) {
        if (group == turn) {
            FRAGMENTS(STAGE)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint top = row0 + turn / 2 * 32;
        uint left = column0 + turn % 2 * 32;
        for (uint step = 0; step < 8; ++step) {
            uint slot = tid + 128 * step;
            uint c_row = top + slot / 32;
            uint c_column = left + slot % 32;
            if (c_row < m && c_column < n)
                c[c_row * n + c_column] = tile_c[slot];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gemm_f32_reference(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[simdgroup_index_in_threadgroup]],
    uint3 block [[threadgroup_position_in_grid]]) {
    threadgroup float tile_a[1024];
    threadgroup float tile_b[1024];
    threadgroup float tile_c[1024];
    gemm_block<float>(a, b, c, m, n, k, tile_a, tile_b, tile_c, tid, group, block);
}

kernel void gemm_f16_reference(
    device const half *a [[buffer(0)]],
    device const half *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[simdgroup_index_in_threadgroup]],
    uint3 block [[threadgroup_position_in_grid]]) {
    threadgroup half tile_a[1024];
    threadgroup half tile_b[1024];
    threadgroup float tile_c[1024];
    gemm_block<half>(a, b, c, m, n, k, tile_a, tile_b, tile_c, tid, group, block);
}
