//! Builds metal-oxide kernel artifacts for the `#[kernel]` and `#[kernels]` macros and for
//! `cargo metal`. Building needs the pinned nightly toolchain with `rustc-dev`, `rust-src`
//! and `llvm-tools`, the metal-oxide compiler and Apple's Metal toolchain. Each build saves
//! its artifact beside the package's sources, so the package builds without them while its
//! kernels are unchanged, and so do its dependents.

mod cache;
mod compile;
mod compiler;
mod configuration;
mod environment;
mod inputs;
mod kernel_crate;
mod metadata;
mod process;
mod recorded;
mod rust;
mod saved;
mod source;
mod state;
mod toolchain;

pub use compile::BuiltKernels;
pub use compiler::Compiler;
pub use kernel_crate::KernelCrate;
pub use process::Error;
pub use source::SourceKernels;

/// The version of Apple's Metal compiler that kernel builds use.
pub fn metal_compiler() -> Result<String, Error> {
    toolchain::Metal::find()
        .map(|metal| metal.version)
        .map_err(Error)
}

/// The textual form of an AIR file, from the Metal toolchain's `metal-opt`.
pub fn disassemble(air: &std::path::Path) -> Result<String, Error> {
    toolchain::Metal::find()
        .and_then(|metal| metal.disassemble(air))
        .map_err(Error)
}

/// Compiles a compiler output directory's MSL with the Metal flags its ABI requires, as kernel
/// builds do, and returns those flags; hidden because only the compiler's tests call it.
#[doc(hidden)]
pub fn compile_msl(directory: &std::path::Path) -> Result<Vec<String>, Error> {
    let compile = || -> process::Result<Vec<String>> {
        let abi = metal_oxide_artifact::Abi::from_json(&std::fs::read_to_string(
            directory.join(metal_oxide_artifact::ArtifactFile::Abi.name()),
        )?)?;
        let flags = toolchain::metal_flags(&abi);
        toolchain::Metal::find()?.compile(directory, &flags)?;
        Ok(flags)
    };
    compile().map_err(Error)
}
