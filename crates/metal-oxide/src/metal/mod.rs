mod access;
mod argument;
mod backend;
mod batch;
mod buffer;
mod capture;
mod classic;
mod completion;
mod constants;
mod device;
mod failure;
mod failures;
mod logging;
#[cfg(all(test, loom))]
mod loom;
mod metal4;
mod module;
mod operation;
mod pipeline;
mod queue;
mod record;
mod sync;

pub use argument::{Argument, ReadArgument, WriteArgument};
pub use batch::Batch;
pub use buffer::{Buffer, Read, Write};
pub use device::Device;
pub use module::Module;
pub use operation::{Launch, Operation, Pending, Sequence, Timed, TimedPending, sequence};
pub use pipeline::Pipeline;

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}
