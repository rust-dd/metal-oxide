#include <metal_stdlib>
using namespace metal;

// Twin of examples/sort: tiles of 512 keys, two per thread of a 256-thread block. Stage k orders
// keys ascending where position & k is 0 and descending elsewhere.
static inline uint lower(uint t, uint j) {
    return (t << 1) - (t & (j - 1));
}

static inline bool out_of_order(uint a, uint b, uint i, uint k) {
    return (a > b) == ((i & k) == 0);
}

static inline void load(device uint *keys, threadgroup uint *tile, uint base, uint lane) {
    tile[lane] = keys[base + lane];
    tile[lane + 256] = keys[base + lane + 256];
}

static inline void merge(threadgroup uint *tile, uint base, uint lane, uint k, uint j) {
    for (; j > 0; j >>= 1) {
        uint i = lower(lane, j);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint a = tile[i];
        uint b = tile[i + j];
        if (out_of_order(a, b, base + i, k)) {
            tile[i] = b;
            tile[i + j] = a;
        }
    }
}

static inline void store(device uint *keys, threadgroup uint *tile, uint base, uint lane) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    keys[base + lane] = tile[lane];
    keys[base + lane + 256] = tile[lane + 256];
}

kernel void sort_local(
    device uint *keys [[buffer(0)]],
    uint lane [[thread_position_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
    threadgroup uint tile[512];
    uint base = block * 512;
    load(keys, tile, base, lane);
    for (uint k = 2; k <= 512; k <<= 1) {
        merge(tile, base, lane, k, k >> 1);
    }
    store(keys, tile, base, lane);
}

kernel void merge_global(
    device uint *keys [[buffer(0)]],
    constant uint &j [[buffer(1)]],
    constant uint &k [[buffer(2)]],
    uint lane [[thread_position_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
    uint i = lower(block * 256 + lane, j);
    uint a = keys[i];
    uint b = keys[i + j];
    if (out_of_order(a, b, i, k)) {
        keys[i] = b;
        keys[i + j] = a;
    }
}

kernel void merge_local(
    device uint *keys [[buffer(0)]],
    constant uint &k [[buffer(1)]],
    uint lane [[thread_position_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
    threadgroup uint tile[512];
    uint base = block * 512;
    load(keys, tile, base, lane);
    merge(tile, base, lane, k, 256);
    store(keys, tile, base, lane);
}
