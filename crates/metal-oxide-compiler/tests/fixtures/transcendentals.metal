#include <metal_stdlib>
using namespace metal;
#ifdef METAL_OXIDE_FAST_MATH
#pragma METAL fp math_mode(relaxed)
#pragma METAL fp contract(fast)
namespace math_functions = fast;
#else
#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
namespace math_functions = precise;
#endif

kernel void transcendentals(
    device const float* input [[buffer(0)]],
    device const float* exponent [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < n) {
        float x = input[i];
        float y = exponent[i];
        out[i * 8] = math_functions::log(x);
        out[i * 8 + 1] = math_functions::exp(x);
        out[i * 8 + 2] = math_functions::sin(x);
        out[i * 8 + 3] = math_functions::cos(x);
        out[i * 8 + 4] = math_functions::pow(x, y);
        out[i * 8 + 5] = math_functions::tanh(x);
        out[i * 8 + 6] = copysign(math_functions::atan(x), x);
        out[i * 8 + 7] = floor(x);
    }
}

kernel void box_muller_step(
    device const float* uniforms [[buffer(0)]],
    device const float* angles [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < n) {
        float z = math_functions::sqrt(-2.0f * math_functions::log(uniforms[i]))
            * math_functions::cos(6.283185307179586f * angles[i]);
        out[i] = 100.0f * math_functions::exp(0.01f + 0.2f * z);
    }
}
