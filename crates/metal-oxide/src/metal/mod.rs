mod access;
mod argument;
mod backend;
mod batch;
mod buffer;
mod classic;
mod completion;
mod constants;
mod device;
mod failure;
#[cfg(all(test, loom))]
mod loom;
mod metal4;
mod module;
mod pipeline;
mod sync;

pub use argument::Argument;
pub use batch::Batch;
pub use buffer::Buffer;
pub use completion::{Submission, SubmissionReport};
pub use device::Device;
pub use module::Module;
pub use pipeline::Pipeline;

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}
