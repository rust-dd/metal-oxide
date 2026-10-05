#include <metal_stdlib>

using namespace metal;

kernel void launch_geometry(
    device uint* out [[buffer(0)]],
    uint3 global_idx [[thread_position_in_grid]],
    uint3 thread_idx [[thread_position_in_threadgroup]],
    uint3 block_idx [[threadgroup_position_in_grid]],
    uint3 block_dim [[threads_per_threadgroup]],
    uint3 grid_dim [[threadgroups_per_grid]]
) {
    uint3 extent = grid_dim * block_dim;
    uint index = global_idx.x + extent.x * (global_idx.y + extent.y * global_idx.z);
    uint base = index * 15;

    out[base] = global_idx.x;
    out[base + 1] = global_idx.y;
    out[base + 2] = global_idx.z;
    out[base + 3] = thread_idx.x;
    out[base + 4] = thread_idx.y;
    out[base + 5] = thread_idx.z;
    out[base + 6] = block_idx.x;
    out[base + 7] = block_idx.y;
    out[base + 8] = block_idx.z;
    out[base + 9] = block_dim.x;
    out[base + 10] = block_dim.y;
    out[base + 11] = block_dim.z;
    out[base + 12] = grid_dim.x;
    out[base + 13] = grid_dim.y;
    out[base + 14] = grid_dim.z;
}
