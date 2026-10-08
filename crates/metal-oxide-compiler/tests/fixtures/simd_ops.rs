#![no_std]
use metal_oxide_device::{WriteBuffer, block_dim, block_idx, kernel, simdgroup, thread_idx};
macro_rules! numeric {
    ($name:ident, $ty:ty) => {
        #[kernel]
        pub unsafe fn $name(out: WriteBuffer<$ty>) {
            let lane = simdgroup::lane_id();
            let value = lane as $ty + 1 as $ty;
            let offset = (block_idx().x * block_dim().x + thread_idx().x) * 8;
            unsafe {
                out.store_unchecked(offset, simdgroup::sum(value));
                out.store_unchecked(offset + 1, simdgroup::min(value));
                out.store_unchecked(offset + 2, simdgroup::max(value));
                out.store_unchecked(offset + 3, simdgroup::inclusive_sum(value));
                out.store_unchecked(offset + 4, simdgroup::exclusive_sum(value));
                out.store_unchecked(offset + 5, simdgroup::shuffle(value, lane));
                out.store_unchecked(offset + 6, simdgroup::shuffle_up(value, 1));
                out.store_unchecked(offset + 7, simdgroup::shuffle_down(value, 0));
            }
        }
    };
}
numeric!(collect_u32, u32);
numeric!(collect_i32, i32);
numeric!(collect_f32, f32);

#[kernel]
pub unsafe fn vote(out: WriteBuffer<u32>) {
    let lane = simdgroup::lane_id();
    let offset = (block_idx().x * block_dim().x + thread_idx().x) * 8;
    unsafe {
        let bits = simdgroup::ballot(lane % 2 == 0);
        out.store_unchecked(offset, bits[0]);
        out.store_unchecked(offset + 1, bits[1]);
        out.store_unchecked(offset + 2, simdgroup::any(lane == 0) as u32);
        out.store_unchecked(offset + 3, simdgroup::all(lane == 0) as u32);
        out.store_unchecked(offset + 4, simdgroup::and(lane + 1));
        out.store_unchecked(offset + 5, simdgroup::or(lane + 1));
        out.store_unchecked(offset + 6, simdgroup::xor(lane + 1));
        out.store_unchecked(offset + 7, simdgroup::shuffle_xor(lane, 0));
    }
}

#[kernel]
pub unsafe fn full_permute(out: WriteBuffer<i32>) {
    let lane = simdgroup::lane_id();
    let offset = thread_idx().x * 3;
    let value = -(lane as i32) - 1;
    unsafe {
        out.store_unchecked(offset, simdgroup::shuffle(value, 0));
        out.store_unchecked(offset + 1, simdgroup::shuffle_down(value, 1));
        out.store_unchecked(offset + 2, simdgroup::shuffle_xor(value, 1));
    }
}
