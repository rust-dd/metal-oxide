//! Bitonic sort of `u32` keys: the kernels and the plan of their dispatches, which the example
//! and the benchmarks run.

/// Ascending bitonic sort of a power-of-two count of keys, at least one tile. Stage `k` orders
/// keys ascending where `position & k` is 0 and descending elsewhere.
#[metal_oxide::kernels]
pub mod bitonic {
    use metal_oxide::device::{
        ReadWriteBuffer, ThreadgroupBuffer, block_idx, thread_idx, threadgroup,
    };

    /// Keys per tile: two for each of a block's 256 threads.
    pub const TILE: u32 = 512;

    type Tile = ThreadgroupBuffer<u32, 512>;

    /// The lower position of the pair `j` apart, a power of two, that thread `t` orders.
    fn lower(t: u32, j: u32) -> u32 {
        (t << 1) - (t & (j - 1))
    }

    /// Whether `a` at position `i` and `b` after it are out of order in stage `k`.
    fn out_of_order(a: u32, b: u32, i: u32, k: u32) -> bool {
        (a > b) == ((i & k) == 0)
    }

    /// Copies the tile at `base` into `tile`; `keys` holds that tile.
    unsafe fn load(keys: ReadWriteBuffer<u32>, tile: Tile, base: u32, lane: u32) {
        // SAFETY: the 256 lanes copy the two halves of a tile that the caller vouches for.
        unsafe {
            tile.store_unchecked(lane, keys.load_unchecked(base + lane));
            tile.store_unchecked(lane + 256, keys.load_unchecked(base + lane + 256));
        }
    }

    /// Runs stage `k` from stride `j`, a power of two up to 256, down to 1 in `tile`; every lane of
    /// the block calls it alike.
    unsafe fn merge(tile: Tile, base: u32, lane: u32, k: u32, mut j: u32) {
        while j > 0 {
            let i = lower(lane, j);
            // SAFETY: the barrier orders the previous step, and strides up to 256 give the lanes
            // disjoint pairs within the tile.
            unsafe {
                threadgroup::barrier();
                let (a, b) = (tile.load_unchecked(i), tile.load_unchecked(i + j));
                if out_of_order(a, b, base + i, k) {
                    tile.store_unchecked(i, b);
                    tile.store_unchecked(i + j, a);
                }
            }
            j >>= 1;
        }
    }

    /// Writes `tile` back to the tile at `base` once the block is done; every lane calls it.
    unsafe fn store(keys: ReadWriteBuffer<u32>, tile: Tile, base: u32, lane: u32) {
        // SAFETY: the barrier orders the last step before each lane writes its two keys back.
        unsafe {
            threadgroup::barrier();
            keys.store_unchecked(base + lane, tile.load_unchecked(lane));
            keys.store_unchecked(base + lane + 256, tile.load_unchecked(lane + 256));
        }
    }

    /// # Safety
    /// `keys` holds one tile per block of the grid, and nothing else accesses it meanwhile.
    #[kernel(block = (256, 1, 1))]
    pub unsafe fn sort_local(keys: ReadWriteBuffer<u32>) {
        let tile = threadgroup::shared::<u32, 512>();
        let (lane, base) = (thread_idx().x, block_idx().x * TILE);
        // SAFETY: the block owns its tile of the caller's keys, and every lane runs each stage;
        // stage `k` starts at stride `k / 2`, at most 256.
        unsafe {
            load(keys, tile, base, lane);
            let mut k = 2;
            while k <= TILE {
                merge(tile, base, lane, k, k >> 1);
                k <<= 1;
            }
            store(keys, tile, base, lane);
        }
    }

    /// # Safety
    /// `keys` holds two keys per grid thread, and `j` is a power of two dividing the thread count.
    #[kernel(block = (256, 1, 1))]
    pub unsafe fn merge_global(keys: ReadWriteBuffer<u32>, j: u32, k: u32) {
        let i = lower(block_idx().x * 256 + thread_idx().x, j);
        // SAFETY: `j` divides the thread count, so each thread's pair is its own and in `keys`.
        unsafe {
            let (a, b) = (keys.load_unchecked(i), keys.load_unchecked(i + j));
            if out_of_order(a, b, i, k) {
                keys.store_unchecked(i, b);
                keys.store_unchecked(i + j, a);
            }
        }
    }

    /// # Safety
    /// `keys` holds one tile per block of the grid, and nothing else accesses it meanwhile.
    #[kernel(block = (256, 1, 1))]
    pub unsafe fn merge_local(keys: ReadWriteBuffer<u32>, k: u32) {
        let tile = threadgroup::shared::<u32, 512>();
        let (lane, base) = (thread_idx().x, block_idx().x * TILE);
        // SAFETY: the block owns its tile of the caller's keys; every lane runs strides 256 to 1.
        unsafe {
            load(keys, tile, base, lane);
            merge(tile, base, lane, k, TILE >> 1);
            store(keys, tile, base, lane);
        }
    }
}

/// One dispatch of the plan, with a block of 256 threads for each tile of the padded keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// `sort_local`: the stages up to a tile.
    SortLocal,
    /// `merge_global`: the step of stride `j` in stage `k`.
    MergeGlobal { j: u32, k: u32 },
    /// `merge_local`: the strides below a tile in stage `k`.
    MergeLocal { k: u32 },
}

/// The count that `n` keys are padded to with `u32::MAX`: the next power of two, at least one
/// tile, or 0 without keys; `n` is at most 2^31.
pub fn padded_len(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let len = n.checked_next_power_of_two().expect("at most 2^31 keys");
    len.max(bitonic::TILE)
}

/// The steps that sort `len` padded keys, in order; none for 0 keys.
pub fn plan(len: u32) -> Vec<Step> {
    if len == 0 {
        return Vec::new();
    }
    assert!(
        len.is_power_of_two() && len >= bitonic::TILE,
        "{len} keys are not padded"
    );
    let tile = bitonic::TILE.trailing_zeros();
    let mut steps = vec![Step::SortLocal];
    for stage in tile + 1..=len.trailing_zeros() {
        let k = 1 << stage;
        let strides = (tile..stage).rev();
        steps.extend(strides.map(|stride| Step::MergeGlobal { j: 1 << stride, k }));
        steps.push(Step::MergeLocal { k });
    }
    steps
}
