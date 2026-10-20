#include <metal_stdlib>
using namespace metal;
kernel void generated(device const uint* input [[buffer(0)]], device uint* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    uint values[3];
    for (uint k=0; k < 3; ++k) values[k] = uint(input[k]);
    uint i = input[3] % 3;
    uint j = input[4] % 3;
    values[i] = values[j];
    output[tid] = uint(values[tid % 3]);
}
