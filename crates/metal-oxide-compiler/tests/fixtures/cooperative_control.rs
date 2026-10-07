#![no_std]

use metal_oxide_device::{WriteBuffer, block_idx, kernel, simdgroup, thread_idx, threadgroup};

unsafe fn synchronize() {
    // SAFETY: the caller keeps all block threads on the same control-flow path.
    unsafe {
        threadgroup::barrier();
    }
}

unsafe fn sum(value: f32) -> f32 {
    // SAFETY: the caller keeps all SIMD lanes on the same control-flow path.
    unsafe { simdgroup::sum(value) }
}

#[kernel(block = (32, 1, 1))]
pub unsafe fn cooperative_exits(output: WriteBuffer<u32>, sums: WriteBuffer<f32>, mode: u32) {
    let shared = threadgroup::shared::<u32, 32>();
    let lane = thread_idx().x;
    let index = block_idx().x * 32 + lane;
    // SAFETY: 32-thread blocks own distinct outputs; every lane initializes its shared cell.
    unsafe {
        shared.store_unchecked(lane, lane + 1);
        synchronize();
        if mode == 9 {
            return;
        }
        let mut i = 0;
        'outer: while i < 4 {
            i += 1;
            let mut j = 0;
            while j < 3 {
                j += 1;
                match mode {
                    0 if j == 2 => continue,
                    1 if i == 2 && j == 2 => break,
                    2 if i == 3 && j == 1 => continue 'outer,
                    3 if i == 4 && j == 2 => break 'outer,
                    _ => {}
                }
                synchronize();
                let value = shared.load_unchecked(lane) + i * 10 + j;
                shared.store_unchecked(lane, value);
                synchronize();
            }
        }
        synchronize();
        output.store_unchecked(index, shared.load_unchecked((lane + 1) & 31));
        sums.store_unchecked(index, sum((lane + 1) as f32));
    }
}
