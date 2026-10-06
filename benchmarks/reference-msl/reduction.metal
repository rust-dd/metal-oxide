#include <metal_stdlib>
using namespace metal;

kernel void reduce(device const float *input [[buffer(0)]], device float *output [[buffer(1)]], constant uint &n [[buffer(2)]], uint lane [[thread_position_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
    threadgroup float values[256];
    uint i = block * 256 + lane;
    values[lane] = i < n ? input[i] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) values[lane] += values[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[block] = values[0];
}
