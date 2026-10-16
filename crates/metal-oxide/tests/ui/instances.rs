#[metal_oxide::kernel(instances())]
pub fn empty<T>(value: T) {}

#[metal_oxide::kernel(instances(same(f32), same(u32)))]
pub fn duplicate<T>(value: T) {}

#[metal_oxide::kernel(instances(missing(f32)))]
pub fn arity<T, const N: usize>(value: T) {}

#[metal_oxide::kernel(instances(concrete(f32)))]
pub fn nongeneric(value: f32) {}

fn main() {}
