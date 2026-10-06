#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[derive(Clone, Copy)]
struct Moments { sum: f32, product: f32 }

fn combine(a: f32, b: f32) -> Moments { Moments { sum: a+b, product: a*b } }

fn scale<const K: u32>(mut value: Moments) -> Moments {
    value.sum *= K as f32;
    value.product += value.sum;
    value
}

#[kernel]
pub unsafe fn record_math(a: ReadBuffer<f32>, b: ReadBuffer<f32>, sum: WriteBuffer<f32>, product: WriteBuffer<f32>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        // SAFETY: all distinct buffers cover n; each active thread owns its output elements.
        unsafe {
            let value = scale::<3>(combine(a.load_unchecked(i),b.load_unchecked(i)));
            sum.store_unchecked(i,value.sum);
            product.store_unchecked(i,value.product);
        }
    }
}
