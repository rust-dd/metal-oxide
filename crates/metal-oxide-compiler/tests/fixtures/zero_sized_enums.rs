#![no_std]

use core::convert::Infallible;
use metal_oxide_device::{WriteBuffer, kernel};

#[derive(Clone, Copy)]
enum Unit {
    Only,
}

// `Never` is uninhabited, so `Choice` is zero-sized as well.
enum Choice {
    Taken,
    Never(Infallible),
}

struct Wrapper<T>(T);

#[inline(never)]
fn wrap(value: Unit) -> Wrapper<Unit> {
    Wrapper(value)
}

#[inline(never)]
fn read(value: &Unit) -> Unit {
    *value
}

#[kernel]
pub unsafe fn zero_sized(out: WriteBuffer<u32>, n: u32) {
    let wrapped = wrap(Unit::Only).0;
    let read = read(&wrapped);
    let folded = (0..n).step_by(3).take(4).fold(read, |acc, _| acc);
    let choice = (Choice::Taken,);
    let chosen = match choice.0 {
        Choice::Never(_) => 2,
        Choice::Taken => 1,
    };
    unsafe { out.store_unchecked(0, matches!(folded, Unit::Only) as u32 + n + chosen * 10) };
}
