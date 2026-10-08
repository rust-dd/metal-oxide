use metal_oxide_device::{
    ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, simdgroup, thread_idx, threadgroup,
};

/// # Safety
/// Disjoint input/prefix buffers cover n; totals covers ceil(n/64).
/// Every block has 64 uniformly participating threads in a one-dimensional grid.
#[kernel(block = (64, 1, 1))]
pub unsafe fn block_scan(
    input: ReadBuffer<u32>,
    prefix: WriteBuffer<u32>,
    totals: WriteBuffer<u32>,
    n: u32,
) {
    let groups = threadgroup::shared::<u32, 64>();
    let thread = thread_idx().x;
    let i = block_idx().x * block_dim().x + thread;
    let lane = simdgroup::lane_id();
    let group = simdgroup::group_id();
    let mut value = 0;
    // SAFETY: padded lanes contribute zero; group leaders initialize totals before the barrier.
    unsafe {
        if i < n {
            value = input.load_unchecked(i);
        }
        let prefix_in_group = simdgroup::exclusive_sum(value);
        let group_total = simdgroup::sum(value);
        if lane == 0 {
            groups.store_unchecked(group, group_total);
        }
        threadgroup::barrier();
        let mut offset = 0;
        let mut preceding = 0;
        while preceding < group {
            offset += groups.load_unchecked(preceding);
            preceding += 1;
        }
        let result = offset + prefix_in_group;
        if i < n {
            prefix.store_unchecked(i, result);
        }
        if thread + 1 == block_dim().x {
            totals.store_unchecked(block_idx().x, result + value);
        }
    }
}

/// # Safety
/// Disjoint local/output buffers cover n; offsets covers ceil(n/64) and contains
/// the exclusive scan of the preceding pass's block totals.
#[kernel(block = (64, 1, 1))]
pub unsafe fn add_offsets(
    local: ReadBuffer<u32>,
    offsets: ReadBuffer<u32>,
    output: WriteBuffer<u32>,
    n: u32,
) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        // SAFETY: the guard bounds the unique output; each block has one initialized offset.
        unsafe {
            output.store_unchecked(
                i,
                local.load_unchecked(i) + offsets.load_unchecked(block_idx().x),
            );
        }
    }
}
