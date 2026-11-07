use std::fmt;

/// The result of a metal-oxide operation.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A runtime check that failed in a kernel thread, reported like a Rust panic.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct KernelFailure {
    /// The kernel's name.
    pub kernel: String,
    /// Position of the failed dispatch among the dispatches pushed into its command buffer.
    pub dispatch: usize,
    /// The panic message of the failed check.
    pub message: String,
    /// The Rust source file of the failed check.
    pub file: String,
    /// The 1-based line of the failed check.
    pub line: u32,
    /// The 1-based column of the failed check.
    pub column: u32,
}

impl fmt::Display for KernelFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "kernel `{}` panicked at {}:{}:{}:\n{}",
            self.kernel, self.file, self.line, self.column, self.message
        )
    }
}

/// Why a metal-oxide operation failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The system has no Metal device.
    DeviceUnavailable,
    /// The device, macOS or native API lacks something that the operation requires.
    UnsupportedDevice(String),
    /// `METAL_OXIDE_CAPTURE` names a GPU trace that Metal cannot record.
    Capture(String),
    /// Metal could not allocate a buffer.
    AllocationFailed {
        /// The requested size.
        bytes: usize,
    },
    /// A buffer exceeds the device's maximum buffer length.
    BufferTooLarge {
        /// The requested size.
        bytes: usize,
        /// The device's maximum buffer length in bytes.
        maximum: usize,
    },
    /// A buffer's byte length overflows the host address range.
    LengthOverflow,
    /// A buffer cannot be used as requested, such as for CPU access while a conflicting
    /// recording is uncommitted.
    InvalidBuffer(&'static str),
    /// A launch configuration does not fit its pipeline or device.
    InvalidLaunch(&'static str),
    /// A launch binds more buffers than Metal's 31 slots.
    TooManyArguments(usize),
    /// The arguments of a launch violate a relation in its kernel's `requires`; nothing was
    /// encoded.
    Contract {
        /// The kernel's name.
        kernel: String,
        /// The violated relation, as written in `requires`.
        relation: String,
    },
    /// A pipeline, module or buffer belongs to another device context.
    DeviceMismatch,
    /// Metal could not create a library from source, a file or bytes.
    Library(String),
    /// A library has no kernel of this name.
    KernelNotFound(String),
    /// Metal could not create a compute pipeline; Apple's GPU compiler can fail for one GPU.
    Pipeline(String),
    /// A command buffer or a native runtime object failed.
    Command(String),
    /// A command buffer ran out of GPU memory.
    OutOfMemory(String),
    /// A dispatch failed a runtime check; the dispatches of its command buffer that wait for
    /// it were skipped.
    KernelFailed(Box<KernelFailure>),
    /// An operation still running when this one committed, whose writes this one reads, failed
    /// a runtime check; this operation's generated kernels were skipped.
    DependencyFailed(Box<KernelFailure>),
    /// A library path is not valid UTF-8.
    InvalidPath,
    /// A kernel artifact failed validation, such as of its ABI version or library digest.
    Artifact(metal_oxide_artifact::Error),
    /// Reading a kernel artifact or library file failed.
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceUnavailable => write!(f, "no Metal device is available"),
            Self::UnsupportedDevice(reason) => write!(f, "unsupported Metal device: {reason}"),
            Self::Capture(reason) => write!(f, "GPU capture failed: {reason}"),
            Self::AllocationFailed { bytes } => {
                write!(f, "failed to allocate {bytes} Metal buffer bytes")
            }
            Self::BufferTooLarge { bytes, maximum } => {
                write!(f, "buffer size {bytes} exceeds limit {maximum}")
            }
            Self::LengthOverflow => write!(f, "buffer length overflows the host address range"),
            Self::InvalidBuffer(reason) => write!(f, "invalid Metal buffer: {reason}"),
            Self::InvalidLaunch(reason) => write!(f, "invalid launch: {reason}"),
            Self::TooManyArguments(count) => {
                write!(f, "{count} arguments exceed the 31 Metal buffer slots")
            }
            Self::Contract { kernel, relation } => {
                write!(f, "kernel {kernel} requires {relation}")
            }
            Self::DeviceMismatch => write!(
                f,
                "pipeline, module, and buffers must belong to the same Metal device context"
            ),
            Self::Library(reason) => write!(f, "Metal library error: {reason}"),
            Self::KernelNotFound(name) => write!(f, "Metal kernel {name:?} was not found"),
            Self::Pipeline(reason) => write!(f, "Metal pipeline error: {reason}"),
            Self::Command(reason) => write!(f, "Metal command failed: {reason}"),
            Self::OutOfMemory(reason) => write!(f, "Metal command ran out of memory: {reason}"),
            Self::KernelFailed(failure) => failure.fmt(f),
            Self::DependencyFailed(failure) => {
                write!(f, "an operation this one reads from failed: {failure}")
            }
            Self::InvalidPath => write!(f, "Metal library path is not valid UTF-8"),
            Self::Artifact(error) => write!(f, "kernel artifact error: {error}"),
            Self::Io(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Artifact(error) => Some(error),
            _ => None,
        }
    }
}

impl From<metal_oxide_artifact::Error> for Error {
    fn from(error: metal_oxide_artifact::Error) -> Self {
        Self::Artifact(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
