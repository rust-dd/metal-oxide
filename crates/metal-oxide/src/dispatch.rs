use crate::{Error, Result};

/// A one-dimensional grid with an optional threadgroup width.
#[derive(Clone, Copy, Debug)]
pub struct Dispatch1d {
    threads: u32,
    group_width: Option<usize>,
}

impl Dispatch1d {
    /// An empty grid is a no-op. Otherwise the pipeline selects the group width.
    pub const fn new(threads: u32) -> Self {
        Self {
            threads,
            group_width: None,
        }
    }

    pub const fn with_group_width(mut self, width: usize) -> Self {
        self.group_width = Some(width);
        self
    }

    pub const fn threads(self) -> u32 {
        self.threads
    }

    /// Validates the requested width against the actual compute pipeline limits.
    pub fn group_width(self, execution_width: usize, maximum: usize) -> Result<usize> {
        if execution_width == 0 || maximum == 0 || execution_width > maximum {
            return Err(Error::InvalidDispatch(
                "invalid pipeline threadgroup limits",
            ));
        }
        let width = self.group_width.unwrap_or(execution_width);
        if width == 0 || width > maximum {
            return Err(Error::InvalidDispatch(
                "threadgroup width must be within pipeline limits",
            ));
        }
        Ok(width)
    }
}
