use crate::diagnostics::{self, Positions};
use metal_oxide_artifact::{ArtifactFile, MathMode, sha256};
use metal_oxide_build::{BuiltKernels, KernelCrate, SourceKernels};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::{io::Write, path::PathBuf};
use syn::{Ident, LitStr};

type Position = (usize, usize);

const CAPTURE: &str = "METAL_OXIDE_CAPTURE";
/// A file that lists the artifacts a build compiled, for `cargo metal inspect`.
const INSPECT: &str = "METAL_OXIDE_INSPECT";
const ARTIFACTS: &str = "METAL_OXIDE_ARTIFACTS";
const COMPILER: &str = "METAL_OXIDE_COMPILER";

/// Appends a compiled artifact to the inspection list, when `cargo metal inspect` asked for one.
fn report(built: &BuiltKernels, item: &str) {
    let Some(list) = std::env::var_os(INSPECT).filter(|_| !built.saved) else {
        return;
    };
    let package = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let line = format!("{}\t{package}\t{item}\n", built.directory.display());
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(list)
        .and_then(|mut file| file.write_all(line.as_bytes()));
}

/// Where the macro's item lies in its source file, from rustc's spans.
pub(crate) struct Location {
    pub(crate) file: PathBuf,
    pub(crate) start: Position,
    pub(crate) end: Position,
    pub(crate) hole: Option<(Position, Position)>,
}

/// What the device build of kernels written in the host crate compiles.
pub(crate) struct Request {
    /// `None` when spans carry no positions, as under rust-analyzer.
    pub(crate) location: Option<Location>,
    pub(crate) positions: Positions,
    pub(crate) item: String,
    pub(crate) prelude: bool,
    pub(crate) math_mode: MathMode,
    pub(crate) dependencies: Vec<String>,
}

/// Whether this expansion builds for the GPU: rust-analyzer expands macros on every edit and
/// needs the launch functions only, and hosts without Metal only check saved artifacts, when
/// `METAL_OXIDE_ARTIFACTS=locked` asks for that.
fn builds() -> bool {
    let metal = cfg!(all(target_os = "macos", target_arch = "aarch64"));
    let locked = std::env::var_os(ARTIFACTS).is_some_and(|value| value == "locked");
    std::env::var_os("RUST_ANALYZER_INTERNALS_DO_NOT_USE").is_none() && (metal || locked)
}

fn package() -> syn::Result<PathBuf> {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| syn::Error::new(Span::call_site(), "kernels are compiled in Cargo builds"))
}

/// Builds the kernels that a request copies from the host crate, or `None` when this
/// expansion does not build.
pub(crate) fn source(request: Request) -> syn::Result<Option<BuiltKernels>> {
    if !builds() {
        return Ok(None);
    }
    let item = request.item.clone();
    let location = request.location.ok_or_else(|| {
        syn::Error::new(
            Span::call_site(),
            "kernels must be written directly in a source file",
        )
    })?;
    SourceKernels {
        package: package()?,
        file: location.file.clone(),
        start: location.start,
        end: location.end,
        hole: location.hole,
        item: request.item,
        prelude: request.prelude,
        math_mode: request.math_mode,
        dependencies: request.dependencies,
        debug_info: std::env::var_os(CAPTURE).is_some(),
    }
    .build()
    .map(|built| {
        report(&built, &item);
        Some(built)
    })
    .map_err(|failure| {
        diagnostics::errors(
            &failure.to_string(),
            &location.file,
            &request.positions,
            Span::call_site(),
        )
    })
}

/// Builds the kernel crate in `path`, relative to the host package, for the module `item`
/// in `file`; without a build, the last build's artifact describes its kernels.
pub(crate) fn kernel_crate(
    path: &LitStr,
    file: Option<PathBuf>,
    item: &str,
    math_mode: MathMode,
) -> syn::Result<Option<BuiltKernels>> {
    let kernel_crate = KernelCrate {
        package: package()?,
        file,
        item: item.to_owned(),
        path: PathBuf::from(path.value()),
        math_mode,
        debug_info: std::env::var_os(CAPTURE).is_some(),
    };
    if !builds() {
        return Ok(kernel_crate.previous());
    }
    kernel_crate
        .build()
        .map(|built| {
            report(&built, item);
            Some(built)
        })
        .map_err(|failure| {
            diagnostics::errors(
                &failure.to_string(),
                &PathBuf::new(),
                &Positions::empty(),
                path.span(),
            )
        })
}

/// The static that embeds a library under `name`, with constants that make Cargo rebuild the
/// crate when the library, its inputs or the variables that steer kernel builds change;
/// without a library the static is empty, which still lets the launch functions compile.
pub(crate) fn library(name: &Ident, built: Option<&BuiltKernels>) -> syn::Result<TokenStream> {
    let tracking = quote! {
        const _: ::core::option::Option<&str> = ::core::option_env!(#CAPTURE);
        const _: ::core::option::Option<&str> = ::core::option_env!(#INSPECT);
        const _: ::core::option::Option<&str> = ::core::option_env!(#ARTIFACTS);
        const _: ::core::option::Option<&str> = ::core::option_env!(#COMPILER);
    };
    let Some(built) = built.filter(|_| builds()) else {
        return Ok(quote! {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            #[doc(hidden)]
            static #name: ::metal_oxide::Library = ::metal_oxide::Library::new("", &[], "");
            #tracking
        });
    };
    let path = |path: PathBuf| path.display().to_string();
    let manifest = path(built.directory.join(ArtifactFile::Manifest.name()));
    let metallib = path(built.directory.join(ArtifactFile::Metallib.name()));
    let tracked = built.tracked.iter().map(|file| path(file.clone()));
    let abi = built
        .abi
        .to_json()
        .map_err(|failure| syn::Error::new(Span::call_site(), failure.to_string()))?;
    let abi = sha256(abi.as_bytes());
    Ok(quote! {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        #[doc(hidden)]
        static #name: ::metal_oxide::Library = ::metal_oxide::Library::new(
            ::core::include_str!(#manifest),
            ::core::include_bytes!(#metallib),
            #abi,
        );
        #(const _: &[::core::primitive::u8] = ::core::include_bytes!(#tracked);)*
        #tracking
    })
}
