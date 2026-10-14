use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

/// A runtime check that failed in a kernel thread, reported like a Rust panic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelFailure {
    pub kernel: String,
    /// Position of the failed dispatch among the dispatches pushed into its command buffer.
    pub dispatch: usize,
    pub message: String,
    pub file: String,
    pub line: u32,
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

#[derive(Debug)]
pub enum Error {
    DeviceUnavailable,
    UnsupportedDevice(String),
    AllocationFailed {
        bytes: usize,
    },
    BufferTooLarge {
        bytes: usize,
        maximum: usize,
    },
    LengthOverflow,
    InvalidBuffer(&'static str),
    InvalidLaunch(&'static str),
    TooManyArguments(usize),
    DeviceMismatch,
    Library(String),
    KernelNotFound(String),
    Pipeline(String),
    Command(String),
    OutOfMemory(String),
    /// A dispatch failed a runtime check; the dispatches of its command buffer that wait for
    /// it were skipped.
    KernelFailed(Box<KernelFailure>),
    /// An operation still running when this one committed, whose writes this one reads, failed
    /// a runtime check; this operation's generated kernels were skipped.
    DependencyFailed(Box<KernelFailure>),
    InvalidPath,
    Artifact(metal_oxide_artifact::Error),
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceUnavailable => write!(f, "no Metal device is available"),
            Self::UnsupportedDevice(reason) => write!(f, "unsupported Metal device: {reason}"),
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
