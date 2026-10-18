#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

// Twin of examples/stencil: one Jacobi sweep of a width x height grid, a 16 x 16 block per tile.
// Interior cells take 0.25 * (((l + r) + u) + d); boundary cells copy their input.
kernel void jacobi(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &width [[buffer(2)]],
    constant uint &height [[buffer(3)]],
    uint2 lane [[thread_position_in_threadgroup]],
    uint2 block [[threadgroup_position_in_grid]]) {
    uint x = block.x * 16 + lane.x;
    uint y = block.y * 16 + lane.y;
    if (x < width && y < height) {
        uint i = y * width + x;
        float value;
        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
            value = input[i];
        } else {
            float l = input[i - 1];
            float r = input[i + 1];
            float u = input[i - width];
            float d = input[i + width];
            value = 0.25f * (((l + r) + u) + d);
        }
        output[i] = value;
    }
}
