use crate::{Error, Result};

/// Dimensions along the x, y, and z axes, following CUDA's dim3 convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub const fn x(x: u32) -> Self {
        Self::new(x, 1, 1)
    }

    pub const fn xy(x: u32, y: u32) -> Self {
        Self::new(x, y, 1)
    }

    pub const fn is_empty(self) -> bool {
        self.x == 0 || self.y == 0 || self.z == 0
    }

    fn volume(self) -> Result<u64> {
        u64::from(self.x)
            .checked_mul(u64::from(self.y))
            .and_then(|xy| xy.checked_mul(u64::from(self.z)))
            .ok_or(Error::InvalidLaunch("dimension volume overflows u64"))
    }
}

/// A grid of blocks, with a fixed thread count per block on each axis.
///
/// Metal executes each block as a threadgroup. The grid counts blocks, not
/// individual threads. An empty grid is a no-op; block dimensions must be positive.
///
/// ```
/// use metal_oxide::{Dim3, LaunchConfig};
/// let config = LaunchConfig {
///     grid: Dim3::x(4),
///     block: Dim3::x(256),
/// };
/// assert_eq!(config.total_threads().unwrap(), 1024);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: Dim3,
    pub block: Dim3,
}

impl LaunchConfig {
    pub const fn new(grid: Dim3, block: Dim3) -> Self {
        Self { grid, block }
    }

    /// Covers a one-dimensional element range with complete blocks.
    ///
    /// Extra threads must be guarded by the kernel's bounds check. Device and
    /// pipeline limits are checked separately when the kernel is launched.
    pub fn for_elements(elements: u32, threads_per_block: u32) -> Result<Self> {
        if threads_per_block == 0 {
            return Err(Error::InvalidLaunch("block dimensions must be positive"));
        }
        let config = Self::new(
            Dim3::x(elements.div_ceil(threads_per_block)),
            Dim3::x(threads_per_block),
        );
        config.total_threads()?;
        Ok(config)
    }

    pub const fn is_empty(self) -> bool {
        self.grid.is_empty()
    }

    /// Counts all launched threads, including padding in complete blocks.
    pub fn total_threads(self) -> Result<u64> {
        if self.is_empty() {
            return Ok(0);
        }
        self.global_size()?.volume()
    }

    /// Checks the pipeline's total block limit and the device's per-axis limits.
    pub fn validate(
        self,
        maximum_threads_per_block: usize,
        maximum_block_dimensions: Dim3,
    ) -> Result<()> {
        if maximum_threads_per_block == 0 {
            return Err(Error::InvalidLaunch(
                "pipeline block limit must be positive",
            ));
        }
        if self.block.is_empty() {
            return Err(Error::InvalidLaunch("block dimensions must be positive"));
        }
        if self.block.x > maximum_block_dimensions.x
            || self.block.y > maximum_block_dimensions.y
            || self.block.z > maximum_block_dimensions.z
        {
            return Err(Error::InvalidLaunch(
                "block dimensions exceed device limits",
            ));
        }
        if self.block.volume()? > maximum_threads_per_block as u64 {
            return Err(Error::InvalidLaunch(
                "threads per block exceed pipeline limit",
            ));
        }
        self.total_threads()?;
        Ok(())
    }

    fn global_size(self) -> Result<Dim3> {
        let axis = |blocks: u32, threads: u32| {
            blocks
                .checked_mul(threads)
                .ok_or(Error::InvalidLaunch("global thread dimensions exceed u32"))
        };
        Ok(Dim3::new(
            axis(self.grid.x, self.block.x)?,
            axis(self.grid.y, self.block.y)?,
            axis(self.grid.z, self.block.z)?,
        ))
    }
}
