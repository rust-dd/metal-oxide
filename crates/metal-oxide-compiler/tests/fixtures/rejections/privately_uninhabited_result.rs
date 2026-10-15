#![no_std]
use metal_oxide_device::kernel;

mod hidden {
    // The private field hides from callers that `Opaque` is uninhabited.
    pub struct Opaque(core::convert::Infallible);

    pub fn make() -> Opaque {
        panic!("never")
    }
}

#[kernel]
pub unsafe fn invalid() {
    let _opaque = hidden::make();
}
