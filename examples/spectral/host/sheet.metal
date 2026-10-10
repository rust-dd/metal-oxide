kernel void sheet_generate(
    device float* dst_real [[buffer(0)]],
    device float* dst_imag [[buffer(1)]],
    device const float* lam [[buffer(2)]],
    device const uint* bit_rev [[buffer(3)]],
    constant uint& cells [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& seed [[buffer(6)]],
    constant ulong& first_cell [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint sheet = tid / cells;
    uint local = tid % cells;
    uint row = local / cols;
    uint col = local % cols;

    float4 u = u01x4((ulong)tid + first_cell, seed);
    float u1 = u.x;
    float u2 = u.y;
    float u3 = u.z;
    float u4 = u.w;
    float r_a = sqrt(-2.0f * log(u1 + 1e-10f));
    float r_b = sqrt(-2.0f * log(u3 + 1e-10f));
    float n_re = r_a * cos(6.28318530718f * u2);
    float n_im = r_b * cos(6.28318530718f * u4);

    float l = lam[local];
    uint dst = sheet * cells + row * cols + bit_rev[col];
    dst_real[dst] = n_re * l;
    dst_imag[dst] = n_im * l;
}
kernel void sheet_transpose(
    device const float* src_real [[buffer(0)]],
    device const float* src_imag [[buffer(1)]],
    device float* dst_real [[buffer(2)]],
    device float* dst_imag [[buffer(3)]],
    device const uint* bit_rev [[buffer(4)]],
    constant uint& cells [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint sheet = tid / cells;
    uint local = tid % cells;
    uint row = local / cols;
    uint col = local % cols;
    uint dst = sheet * cells + col * rows + bit_rev[row];
    dst_real[dst] = src_real[tid];
    dst_imag[dst] = src_imag[tid];
}
kernel void sheet_extract(
    device const float* freq_real [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& cells [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant float& r [[buffer(6)]],
    constant float& corr [[buffer(7)]],
    constant uint& seed [[buffer(8)]],
    constant ulong& corr_cell [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    uint per = m * n;
    uint sheet = tid / per;
    uint local = tid % per;
    uint i = local / n;
    uint j = local % n;
    uint base = sheet * cells;
    float value = freq_real[base + j * rows + i] - freq_real[base];

    float4 u = u01x4(corr_cell + (ulong)sheet, seed);
    float u1 = u.x;
    float u2 = u.y;
    float u3 = u.z;
    float u4 = u.w;
    float z1 = sqrt(-2.0f * log(u1 + 1e-10f)) * cos(6.28318530718f * u2);
    float z2 = sqrt(-2.0f * log(u3 + 1e-10f)) * cos(6.28318530718f * u4);
    float ty = r * float(i + 1u) / float(m);
    float tx = r * float(j + 1u) / float(n);
    output[tid] = value + corr * (ty * z1 + tx * z2);
}
