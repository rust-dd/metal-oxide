#include <metal_stdlib>
using namespace metal;

inline uint classify_value(int value) {
    uint tag;
    switch (value) {
        case (-2147483647 - 1): tag = 1; break;
        case -7: case -3: tag = 2; break;
        case 0: tag = 3; break;
        case 1: tag = 4; break;
        case 2: tag = 5; break;
        case 2147483647: tag = 8; break;
        default: tag = value >= 9 && value <= 12 ? 7 : 9;
    }
    uint byte;
    switch (uchar(value)) {
        case 0: byte = 1; break;
        case 1: case 7: byte = 2; break;
        case 255: byte = 3; break;
        default: byte = 4;
    }
    uint short_value;
    switch (ushort(value)) {
        case 0: short_value = 1; break;
        case 1: case 7: short_value = 2; break;
        case 65535: short_value = 3; break;
        default: short_value = 4;
    }
    uint unsigned_value;
    switch (uint(value)) {
        case 0: unsigned_value = 1; break;
        case 1: case 7: unsigned_value = 2; break;
        case 0xffffffffu: unsigned_value = 3; break;
        default: unsigned_value = 4;
    }
    return tag | (byte << 8) | (short_value << 16) | (unsigned_value << 24);
}

inline uint search_value(uint limit) {
    uint total = 0;
    uint i = 0;
    while (i < limit) {
        ++i;
        if ((i & 3) == 0) continue;
        uint j = 0;
        bool next_outer = false;
        bool done = false;
        while (j < 7) {
            ++j;
            if (j == 2) continue;
            if (i == 3 && j == 3) break;
            if (i == 5 && j == 4) { next_outer = true; break; }
            if (i == 7 && j == 5) { done = true; break; }
            if (limit == 9 && j == 3) return total + 1000;
            total += i * 10 + j;
        }
        if (done) break;
        if (next_outer) continue;
        total += 100;
    }
    return total;
}

inline uint nested_value(uint mode) {
    uint sum = 0;
    uint i = 0;
    while (i < 3) {
        ++i;
        uint j = 0;
        bool next_outer = false;
        bool done = false;
        while (j < 3) {
            ++j;
            uint k = 0;
            while (true) {
                ++k;
                if (k == 2) continue;
                if (k >= 4) break;
                if (mode == 0 && i == 2 && j == 2 && k == 1) { done = true; break; }
                if (mode == 1 && j == 2 && k == 1) { next_outer = true; break; }
                if (mode == 2 && k == 3) break;
                if (mode == 3 && i == 3) return sum + 1000;
                sum += i * 100 + j * 10 + k;
            }
            if (done || next_outer) break;
        }
        if (done) break;
        if (next_outer) continue;
    }
    return sum;
}

kernel void classify(
    device const int *input [[buffer(0)]], device uint *output [[buffer(1)]],
    constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]
) {
    if (i < n) output[i] = classify_value(input[i]);
}

kernel void bounded_search(
    device const uint *input [[buffer(0)]], device uint *output [[buffer(1)]],
    constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]
) {
    if (i < n) output[i] = search_value(input[i]);
}

kernel void nested_exits(
    device const uint *input [[buffer(0)]], device uint *output [[buffer(1)]],
    constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]
) {
    if (i < n) output[i] = nested_value(input[i]);
}

kernel void cooperative_exits(
    device uint *output [[buffer(0)]], device float *sums [[buffer(1)]],
    constant uint &mode [[buffer(2)]], uint lane [[thread_position_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]
) {
    threadgroup uint shared[32];
    shared[lane] = lane + 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (mode == 9) return;
    uint i = 0;
    while (i < 4) {
        ++i;
        uint j = 0;
        bool next_outer = false;
        bool done = false;
        while (j < 3) {
            ++j;
            if (mode == 0 && j == 2) continue;
            if (mode == 1 && i == 2 && j == 2) break;
            if (mode == 2 && i == 3 && j == 1) { next_outer = true; break; }
            if (mode == 3 && i == 4 && j == 2) { done = true; break; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            shared[lane] += i * 10 + j;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (done) break;
        if (next_outer) continue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint index = block * 32 + lane;
    output[index] = shared[(lane + 1) & 31];
    sums[index] = simd_sum(float(lane + 1));
}
