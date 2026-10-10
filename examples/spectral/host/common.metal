#include <metal_stdlib>
using namespace metal;

inline float u01(uint x) {
    return (float(x >> 8) + 0.5f) / 16777216.0f;
}
inline ulong sm64(ulong x) {
    x += 0x9e3779b97f4a7c15UL;
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9UL;
    x ^= x >> 27; x *= 0x94d049bb133111ebUL;
    x ^= x >> 31;
    return x;
}
inline float4 u01x4(ulong cell, uint seed) {
    ulong k = cell ^ ((ulong)seed * 0x9e3779b97f4a7c15UL);
    ulong h1 = sm64(k);
    ulong h2 = sm64(k ^ 0xd1b54a32d192ed03UL);
    return float4(u01((uint)h1), u01((uint)(h1 >> 32)),
                  u01((uint)h2), u01((uint)(h2 >> 32)));
}

kernel void fft_butterfly(
    device float* real [[buffer(0)]],
    device float* imag [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& half_stride [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint butterflies_per_batch = n / 2;
    uint batch = tid / butterflies_per_batch;
    uint local_tid = tid % butterflies_per_batch;
    uint stride = half_stride * 2;
    uint group = local_tid / half_stride;
    uint pos = local_tid % half_stride;
    uint base = batch * n;
    uint i = base + group * stride + pos;
    uint j = i + half_stride;

    float angle = -2.0f * 3.14159265358979323846f * float(pos) / float(stride);
    float tw_r = cos(angle);
    float tw_i = sin(angle);

    float tr = real[j] * tw_r - imag[j] * tw_i;
    float ti = real[j] * tw_i + imag[j] * tw_r;
    float ar = real[i];
    float ai = imag[i];

    real[i] = ar + tr;
    imag[i] = ai + ti;
    real[j] = ar - tr;
    imag[j] = ai - ti;
}
