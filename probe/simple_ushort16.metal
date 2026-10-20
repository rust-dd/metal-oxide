#include <metal_stdlib>
using namespace metal;
kernel void generated(device const uint* input [[buffer(0)]], device uint* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    ushort values[16];
    for (uint k=0; k < 16; ++k) values[k] = ushort(input[k]);
    uint i = input[16] % 16;
    uint j = input[17] % 16;
    values[i] = values[j];
    output[tid] = uint(values[tid % 16]);
}
