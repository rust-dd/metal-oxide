mod argument;
mod batch;
mod buffer;
mod classic;
mod completion;
mod device;
mod module;
mod pipeline;

pub use argument::Argument;
pub use batch::Batch;
pub use buffer::Buffer;
pub use completion::Submission;
pub use device::Device;
pub use module::Module;
pub use pipeline::Pipeline;

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}
