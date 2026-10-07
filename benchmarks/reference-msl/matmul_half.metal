#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

kernel void matmul_half_reference(
    device const half *a [[buffer(0)]],
    device const half *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &inner [[buffer(5)]],
    uint3 lane [[thread_position_in_threadgroup]],
    uint3 block [[threadgroup_position_in_grid]]) {
    threadgroup float tile_a[256];
    threadgroup float tile_b[256];
    uint row = block.y * 16 + lane.y;
    uint column = block.x * 16 + lane.x;
    uint slot = lane.y * 16 + lane.x;
    float result = 0.0f;
    for (uint base = 0; base < inner; base += 16) {
        float x = 0.0f;
        float y = 0.0f;
        if (row < rows && base + lane.x < inner)
            x = float(a[row * inner + base + lane.x]);
        if (column < columns && base + lane.y < inner)
            y = float(b[(base + lane.y) * columns + column]);
        tile_a[slot] = x;
        tile_b[slot] = y;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; ++k)
            result += tile_a[lane.y * 16 + k] * tile_b[k * 16 + lane.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < rows && column < columns)
        output[row * columns + column] = result;
}
