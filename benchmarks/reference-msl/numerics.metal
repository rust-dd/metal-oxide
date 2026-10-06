#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

kernel void dot(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *output [[buffer(2)]], constant uint &n [[buffer(3)]], uint lane [[thread_index_in_simdgroup]], uint group [[simdgroup_index_in_threadgroup]], uint groups [[simdgroups_per_threadgroup]], uint tid [[thread_position_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
    uint i = block * 256 + tid;
    float value = i < n ? a[i] * b[i] : 0.0f;
    float sum = simd_sum(value);
    if (lane == 0) output[block * groups + group] = sum;
}

kernel void matmul(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *output [[buffer(2)]], constant uint &rows [[buffer(3)]], constant uint &columns [[buffer(4)]], constant uint &inner [[buffer(5)]], uint2 tid [[thread_position_in_threadgroup]], uint2 block [[threadgroup_position_in_grid]]) {
    threadgroup float tile_a[256];
    threadgroup float tile_b[256];
    uint row = block.y * 16 + tid.y;
    uint column = block.x * 16 + tid.x;
    uint slot = tid.y * 16 + tid.x;
    float result = 0.0f;
    for (uint base = 0; base < inner; base += 16) {
        tile_a[slot] = row < rows && base + tid.x < inner ? a[row * inner + base + tid.x] : 0.0f;
        tile_b[slot] = column < columns && base + tid.y < inner ? b[(base + tid.y) * columns + column] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; ++k) result += tile_a[tid.y * 16 + k] * tile_b[k * 16 + tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < rows && column < columns) output[row * columns + column] = result;
}
