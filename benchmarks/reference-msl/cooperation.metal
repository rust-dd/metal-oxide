#include <metal_stdlib>
using namespace metal;

kernel void histogram(device const uint *input [[buffer(0)]], device atomic_uint *bins [[buffer(1)]], constant uint &n [[buffer(2)]], uint tid [[thread_position_in_threadgroup]], uint block [[threadgroup_position_in_grid]], uint block_size [[threads_per_threadgroup]]) {
    threadgroup atomic_uint shared[16];
    uint index = block * block_size + tid;
    if (tid < 16) atomic_store_explicit(&shared[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (index < n) atomic_fetch_add_explicit(&shared[input[index] & 15u], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) atomic_fetch_add_explicit(&bins[tid], atomic_load_explicit(&shared[tid], memory_order_relaxed), memory_order_relaxed);
}

kernel void block_scan(device const uint *input [[buffer(0)]], device uint *prefix [[buffer(1)]], device uint *totals [[buffer(2)]], constant uint &n [[buffer(3)]], uint tid [[thread_position_in_threadgroup]], uint block [[threadgroup_position_in_grid]], uint block_size [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint groups[256];
    uint index = block * block_size + tid;
    uint value = index < n ? input[index] : 0u;
    uint prefix_in_group = simd_prefix_exclusive_sum(value);
    uint group_total = simd_sum(value);
    if (lane == 0) groups[group] = group_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint offset = 0u;
    for (uint preceding = 0; preceding < group; ++preceding) offset += groups[preceding];
    uint result = offset + prefix_in_group;
    if (index < n) prefix[index] = result;
    if (tid + 1 == block_size) totals[block] = result + value;
}

kernel void add_offsets(device const uint *local [[buffer(0)]], device const uint *offsets [[buffer(1)]], device uint *output [[buffer(2)]], constant uint &n [[buffer(3)]], uint tid [[thread_position_in_threadgroup]], uint block [[threadgroup_position_in_grid]], uint block_size [[threads_per_threadgroup]]) {
    uint index = block * block_size + tid;
    if (index < n) output[index] = local[index] + offsets[block];
}

kernel void mark(device const uint *input [[buffer(0)]], device uint *flags [[buffer(1)]], constant uint &n [[buffer(2)]], uint tid [[thread_position_in_threadgroup]], uint block [[threadgroup_position_in_grid]], uint block_size [[threads_per_threadgroup]]) {
    uint index = block * block_size + tid;
    if (index < n) flags[index] = input[index] & 1u;
}

kernel void scatter(device const uint *input [[buffer(0)]], device const uint *flags [[buffer(1)]], device const uint *prefix [[buffer(2)]], device uint *output [[buffer(3)]], device uint *count [[buffer(4)]], constant uint &n [[buffer(5)]], uint tid [[thread_position_in_threadgroup]], uint block [[threadgroup_position_in_grid]], uint block_size [[threads_per_threadgroup]]) {
    uint index = block * block_size + tid;
    if (index < n) {
        if (flags[index] != 0u) output[prefix[index]] = input[index];
        if (index + 1u == n) count[0] = prefix[index] + flags[index];
    }
}
