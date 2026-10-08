#![no_std]
use metal_oxide_device::{
    AtomicBuffer, ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, simdgroup, thread_idx,
    threadgroup,
};

/// # Safety
/// A 1D launch covers n inputs; bins has sixteen initialized atomic elements.
/// Inputs and bins are disjoint. Every block has 256 threads.
#[kernel(block = (256, 1, 1))]
pub unsafe fn histogram(input: ReadBuffer<u32>, bins: AtomicBuffer<u32>, n: u32) {
    let shared = threadgroup::shared_atomic::<u32, 16>();
    let thread = thread_idx().x;
    let index = block_idx().x * block_dim().x + thread;
    // SAFETY: complete blocks initialize every bin before contended atomic access.
    unsafe {
        if thread < 16 {
            shared.store_relaxed(thread, 0);
        }
        threadgroup::barrier();
        if index < n {
            shared.fetch_add_relaxed(input.load_unchecked(index) & 15, 1);
        }
        threadgroup::barrier();
        if thread < 16 {
            bins.fetch_add_relaxed(thread, shared.load_relaxed(thread));
        }
    }
}

/// # Safety
/// Disjoint input/prefix buffers cover n, and totals covers ceil(n/256).
/// The launch is 1D with complete 256-thread blocks and uniform participation.
#[kernel(block = (256, 1, 1))]
pub unsafe fn block_scan(
    input: ReadBuffer<u32>,
    prefix: WriteBuffer<u32>,
    totals: WriteBuffer<u32>,
    n: u32,
) {
    let groups = threadgroup::shared::<u32, 256>();
    let thread = thread_idx().x;
    let index = block_idx().x * block_dim().x + thread;
    let lane = simdgroup::lane_id();
    let group = simdgroup::group_id();
    let mut value = 0;
    // SAFETY: padded threads contribute zero; each group leader initializes its own total.
    unsafe {
        if index < n {
            value = input.load_unchecked(index);
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
        if index < n {
            prefix.store_unchecked(index, result);
        }
        if thread + 1 == block_dim().x {
            totals.store_unchecked(block_idx().x, result + value);
        }
    }
}

/// # Safety
/// Input/output cover n and are disjoint; offsets covers ceil(n/256).
/// Offsets contains the exclusive scan of the preceding pass's block totals.
#[kernel(block = (256, 1, 1))]
pub unsafe fn add_offsets(
    local: ReadBuffer<u32>,
    offsets: ReadBuffer<u32>,
    output: WriteBuffer<u32>,
    n: u32,
) {
    let index = block_idx().x * block_dim().x + thread_idx().x;
    // SAFETY: the index guard and block-total allocation cover both reads and the unique write.
    unsafe {
        if index < n {
            output.store_unchecked(
                index,
                local.load_unchecked(index) + offsets.load_unchecked(block_idx().x),
            );
        }
    }
}

/// # Safety
/// Input/flags are disjoint n-element buffers with a 1D launch.
#[kernel]
pub unsafe fn mark(input: ReadBuffer<u32>, flags: WriteBuffer<u32>, n: u32) {
    let index = block_idx().x * block_dim().x + thread_idx().x;
    // SAFETY: every valid index has one writer; the predicate selects odd values.
    unsafe {
        if index < n {
            flags.store_unchecked(index, input.load_unchecked(index) & 1);
        }
    }
}

/// # Safety
/// Input/flags/prefix cover n; prefix is the exclusive scan of flags in {0,1}.
/// Output covers n, count covers one, and all buffers are disjoint.
#[kernel]
pub unsafe fn scatter(
    input: ReadBuffer<u32>,
    flags: ReadBuffer<u32>,
    prefix: ReadBuffer<u32>,
    output: WriteBuffer<u32>,
    count: WriteBuffer<u32>,
    n: u32,
) {
    let index = block_idx().x * block_dim().x + thread_idx().x;
    // SAFETY: selected indices have distinct prefix positions; the last input owns count[0].
    unsafe {
        if index < n {
            let selected = flags.load_unchecked(index);
            let position = prefix.load_unchecked(index);
            if selected != 0 {
                output.store_unchecked(position, input.load_unchecked(index));
            }
            if index + 1 == n {
                count.store_unchecked(0, position + selected);
            }
        }
    }
}
