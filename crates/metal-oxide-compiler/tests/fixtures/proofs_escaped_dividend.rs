#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn escaped_dividend(out: WriteBuffer<i32>, dividend: i32, divisor: i32, exponent: u32) {
    let mut values = [dividend, 1, 2];
    // The index call runs between the overflow check and the remainder.
    values[(exponent.wrapping_pow(exponent) % 3) as usize] = values[0] % divisor;
    for value in &mut values {
        *value = value.wrapping_add(1);
    }
    for (index, value) in values.into_iter().enumerate() {
        unsafe { out.store_unchecked(index as u32, value) };
    }
}
