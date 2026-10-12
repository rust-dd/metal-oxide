//! Builds metal-oxide kernel artifacts from Cargo build scripts and tools.
//!
//! A host package that embeds its kernels keeps the artifact in its sources and calls
//! [`Artifact::update`] from `build.rs`:
//!
//! ```no_run
//! use metal_oxide_build::{Artifact, Error, MathMode};
//!
//! fn main() -> Result<(), Error> {
//!     Artifact::new("../kernels/Cargo.toml", "src/artifact")
//!         .math_mode(MathMode::Fast)
//!         .update()
//! }
//! ```
//!
//! The host then includes `src/artifact/manifest.json`, `kernels.metallib` and
//! `bindings.rs`. Rebuilding an artifact needs the pinned nightly toolchain with
//! `rustc-dev`, `rust-src` and `llvm-tools`, the metal-oxide compiler and Apple's Metal
//! toolchain; checking an unchanged artifact needs none of them.

mod artifact;
mod cache;
mod compile;
mod compiler;
mod inputs;
mod metadata;
mod process;
mod recorded;
mod rust;
mod toolchain;

pub use artifact::Artifact;
pub use compile::Kernels;
pub use compiler::Compiler;
pub use metal_oxide_artifact::MathMode;
pub use process::Error;

/// The version of Apple's Metal compiler that kernel builds use.
pub fn metal_compiler() -> Result<String, Error> {
    toolchain::Metal::find()
        .map(|metal| metal.version)
        .map_err(Error)
}
