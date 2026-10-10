use objc2_foundation::NSError;
use objc2_metal::{MTL4CommandQueueError, MTLCommandBufferError};

#[derive(Clone)]
pub(super) enum CommandFailure {
    OutOfMemory(String),
    Other(String),
}

impl CommandFailure {
    pub(super) fn from_native(error: &NSError) -> Self {
        // Domain names avoid linking Metal 4's new global on older supported macOS versions.
        let memory = match error.domain().to_string().as_str() {
            "MTLCommandBufferErrorDomain" => {
                error.code() == MTLCommandBufferError::OutOfMemory.0 as isize
            }
            "MTL4CommandQueueErrorDomain" => error.code() == MTL4CommandQueueError::OutOfMemory.0,
            _ => false,
        };
        let description = error.localizedDescription().to_string();
        if memory {
            Self::OutOfMemory(description)
        } else {
            Self::Other(description)
        }
    }
}

impl From<CommandFailure> for crate::Error {
    fn from(failure: CommandFailure) -> Self {
        match failure {
            CommandFailure::OutOfMemory(reason) => Self::OutOfMemory(reason),
            CommandFailure::Other(reason) => Self::Command(reason),
        }
    }
}
