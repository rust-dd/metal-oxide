use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    DeviceUnavailable,
    UnsupportedDevice(String),
    AllocationFailed { bytes: usize },
    BufferTooLarge { bytes: usize, maximum: usize },
    LengthOverflow,
    InvalidLaunch(&'static str),
    TooManyArguments(usize),
    DeviceMismatch,
    Library(String),
    KernelNotFound(String),
    Pipeline(String),
    Command(String),
    InvalidPath,
    Artifact(metal_oxide_artifact::Error),
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceUnavailable => write!(f, "no Metal device is available"),
            Self::UnsupportedDevice(name) => write!(f, "device {name} requires unified memory"),
            Self::AllocationFailed { bytes } => {
                write!(f, "failed to allocate {bytes} Metal buffer bytes")
            }
            Self::BufferTooLarge { bytes, maximum } => {
                write!(f, "buffer size {bytes} exceeds limit {maximum}")
            }
            Self::LengthOverflow => write!(f, "buffer length overflows the host address range"),
            Self::InvalidLaunch(reason) => write!(f, "invalid launch: {reason}"),
            Self::TooManyArguments(count) => {
                write!(f, "{count} arguments exceed the 31 Metal buffer slots")
            }
            Self::DeviceMismatch => write!(
                f,
                "pipeline, module, and buffers must belong to the same Metal device"
            ),
            Self::Library(reason) => write!(f, "Metal library error: {reason}"),
            Self::KernelNotFound(name) => write!(f, "Metal kernel {name:?} was not found"),
            Self::Pipeline(reason) => write!(f, "Metal pipeline error: {reason}"),
            Self::Command(reason) => write!(f, "Metal command failed: {reason}"),
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
