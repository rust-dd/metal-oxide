#include <metal_stdlib>

using namespace metal;

kernel void vec_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
