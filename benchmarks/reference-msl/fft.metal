#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

// Twin of examples/fft: the radix-2 Stockham stage of half_span on every row of n values, a
// block per 256 butterflies of a row along x and a row along y, with the kernel's float order.
kernel void fft_stage(
    device const float *input_re [[buffer(0)]],
    device const float *input_im [[buffer(1)]],
    device float *output_re [[buffer(2)]],
    device float *output_im [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &half_span [[buffer(5)]],
    uint2 lane [[thread_position_in_threadgroup]],
    uint2 block [[threadgroup_position_in_grid]]) {
    uint half_n = n >> 1;
    uint butterfly = block.x * 256 + lane.x;
    if (butterfly < half_n) {
        uint row = block.y * n;
        uint j = butterfly & (half_span - 1);
        float cosine;
        float sine = precise::sincos(-2.0f * M_PI_F * float(j) / float(half_span << 1), cosine);
        uint input = row + butterfly;
        uint output = row + (butterfly << 1) - j;
        float a_re = input_re[input];
        float a_im = input_im[input];
        float b_re = input_re[input + half_n];
        float b_im = input_im[input + half_n];
        float t_re = b_re * cosine - b_im * sine;
        float t_im = b_re * sine + b_im * cosine;
        output_re[output] = a_re + t_re;
        output_im[output] = a_im + t_im;
        output_re[output + half_span] = a_re - t_re;
        output_im[output + half_span] = a_im - t_im;
    }
}
