#[metal_oxide::kernels]
mod kernels {
    use metal_oxide::device::{ReadBuffer, WriteBuffer};

    pub struct Pair {
        pub first: u32,
        pub second: u32,
    }

    #[kernel]
    pub unsafe fn sums(pairs: ReadBuffer<Pair>, out: WriteBuffer<u32>) {}
}

fn main() {}
