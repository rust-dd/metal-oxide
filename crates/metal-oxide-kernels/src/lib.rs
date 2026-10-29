//! The `#[kernel]` and `#[kernels]` attributes that `metal-oxide` re-exports. They compile
//! kernels with the metal-oxide compiler at build time and replace each kernel with a function
//! that launches it on a `Device`.

mod abi;
mod device;
mod diagnostics;
mod instances;
mod launch;
mod mirror;
mod options;
mod values;

use device::{Location, Request};
use diagnostics::Positions;
use launch::Kernel;
use mirror::Mirrors;
use options::{KernelOptions, ModuleOptions};
use proc_macro::{Span, TokenStream};
use proc_macro2::TokenStream as Tokens;
use quote::{format_ident, quote};
use syn::{Attribute, Item, ItemFn, ItemMod, LitStr, Meta, ext::IdentExt};

/// Compiles a function as a Metal kernel and replaces it with a function of the same name and
/// visibility that launches it: `vec_add(&device, config, &a, &b, &mut out)` returns a
/// `Launch`. `block = (X, Y, Z)` fixes the block shape and `requires(...)` lists relations
/// that every launch checks. `instances(name(Type, CONST), ...)` generates a named launch
/// function for each listed instance of a generic kernel.
///
/// The kernel sees the device prelude, `core`, items it defines and `no_std` crates by full
/// path; kernels that share types or helpers go in a `#[kernels]` module.
#[proc_macro_attribute]
pub fn kernel(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let source = Source::new(&item, false);
    expand(standalone(attributes.into(), item.into(), source))
}

/// Compiles the `#[kernel]` functions of an inline module as one Metal library. The module's
/// other items stay on the host too, and its `#[repr(C)]` records that kernels take become
/// `GpuValue`s. `math = "fast"` relaxes floating-point math; `dependencies(...)` names the
/// `no_std` crates the kernels use.
///
/// With `path = "../kernels"` the module stays empty and launches the kernels of that kernel
/// crate instead, with a host struct for each record they take.
#[proc_macro_attribute]
pub fn kernels(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let source = Source::new(&item, true);
    expand(module(attributes.into(), item.into(), source))
}

fn expand(result: syn::Result<Tokens>) -> TokenStream {
    result.unwrap_or_else(syn::Error::into_compile_error).into()
}

/// Where the macro's item is in its source file, read before the tokens lose their positions.
struct Source {
    location: Option<Location>,
    positions: Positions,
}

impl Source {
    /// `hole` leaves the attribute itself out of the device source.
    fn new(item: &TokenStream, hole: bool) -> Self {
        Self {
            location: location(item, hole),
            positions: Positions::new(item),
        }
    }

    fn request(self, item: String, prelude: bool, options: ModuleOptions) -> Request {
        Request {
            location: self.location,
            positions: self.positions,
            item,
            prelude,
            math_mode: options.math_mode,
            dependencies: options.dependencies,
        }
    }
}

fn location(item: &TokenStream, hole: bool) -> Option<Location> {
    let call = Span::call_site();
    let file = call.local_file()?;
    let mut tokens = item.clone().into_iter();
    let first = tokens.next()?.span();
    let last = tokens.last().map_or(first, |token| token.span());
    let position = |span: Span| (span.line(), span.column());
    let call_range = (position(call.start()), position(call.end()));
    Some(Location {
        file,
        start: position(first.start()).min(call_range.0),
        end: position(last.end()),
        hole: hole.then_some(call_range),
    })
}

/// The library static for kernels written in the host crate, after checking the build
/// against the signatures; on failure an empty static and the errors, so the host items still
/// compile and the errors stay on their tokens.
fn source_library(library: &syn::Ident, request: Request, kernels: &[Kernel]) -> Tokens {
    let built = device::source(request).and_then(|built| {
        if let Some(built) = &built {
            abi::check(kernels, &built.abi, proc_macro2::Span::call_site())?;
        }
        Ok(built)
    });
    match built.and_then(|built| device::library(library, built.as_ref())) {
        Ok(tokens) => tokens,
        Err(error) => {
            let empty = device::library(library, None).unwrap_or_default();
            let error = error.into_compile_error();
            quote!(#empty #error)
        }
    }
}

fn standalone(attributes: Tokens, item: Tokens, source: Source) -> syn::Result<Tokens> {
    let options = KernelOptions::parse(attributes)?;
    let function = syn::parse2::<ItemFn>(item)?;
    let kernels = instances::kernels(&function, options)?;
    let name = function.sig.ident.unraw();
    let library = format_ident!("__METAL_OXIDE_{}", name.to_string().to_uppercase());
    let hidden = format_ident!("__metal_oxide_kernel_{name}");
    let defaults = ModuleOptions {
        math_mode: Default::default(),
        dependencies: Vec::new(),
        path: None,
    };
    let request = source.request(name.to_string(), true, defaults);
    let library_static = source_library(&library, request, &kernels);
    let launchers = kernels.iter().map(|kernel| kernel.launcher(&library));
    Ok(quote! {
        #[doc(hidden)]
        #[allow(dead_code, non_snake_case)]
        mod #hidden {
            use super::*;
            #function
        }
        #library_static
        #(#launchers)*
    })
}

/// A function that loads the module's library on a device, for pipelines that code encodes
/// itself.
fn module_function(library: &syn::Ident) -> Tokens {
    quote! {
        /// This module's kernel library on `device`, for pipelines that code encodes itself.
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        #[allow(dead_code)]
        pub fn module(device: &::metal_oxide::Device) -> ::metal_oxide::Result<::metal_oxide::Module> {
            #library.module(device)
        }
    }
}

fn module(attributes: Tokens, item: Tokens, source: Source) -> syn::Result<Tokens> {
    let mut options = ModuleOptions::parse(attributes)?;
    let mut module = syn::parse2::<ItemMod>(item)?;
    let Some((_, items)) = module.content.take() else {
        return Err(syn::Error::new_spanned(
            &module,
            "`#[kernels]` applies to an inline module",
        ));
    };
    let ItemMod {
        attrs,
        vis,
        unsafety,
        mod_token,
        ident,
        ..
    } = &module;
    let body = match options.path.take() {
        Some(path) if items.is_empty() => {
            let location = source
                .location
                .map(|location| (location.file, location.start));
            kernel_crate(&path, location, ident, options)?
        }
        Some(path) => {
            return Err(syn::Error::new_spanned(
                path,
                "a module that launches a kernel crate's kernels is empty",
            ));
        }
        None => written(items, ident, options, source)?,
    };
    Ok(quote! {
        #(#attrs)*
        #vis #unsafety #mod_token #ident {
            #body
        }
    })
}

/// The body of a module whose kernels are written in it.
fn written(
    items: Vec<Item>,
    ident: &syn::Ident,
    options: ModuleOptions,
    source: Source,
) -> syn::Result<Tokens> {
    let mut kernels = Vec::new();
    let mut functions = Vec::new();
    let mut others = Vec::new();
    for item in items {
        let Item::Fn(mut function) = item else {
            others.push(item);
            continue;
        };
        let Some(index) = function.attrs.iter().position(is_kernel) else {
            others.push(Item::Fn(function));
            continue;
        };
        let options = match &function.attrs.remove(index).meta {
            Meta::Path(_) => KernelOptions::default(),
            Meta::List(list) => KernelOptions::parse(list.tokens.clone())?,
            Meta::NameValue(value) => {
                return Err(syn::Error::new_spanned(value, "expected `#[kernel(...)]`"));
            }
        };
        kernels.extend(instances::kernels(&function, options)?);
        functions.push(function);
    }
    if kernels.is_empty() {
        return Err(syn::Error::new_spanned(
            ident,
            "a `#[kernels]` module defines `#[kernel]` functions or names a kernel crate",
        ));
    }
    let values = values::gpu_values(&others, &kernels)?;
    let library = format_ident!("__METAL_OXIDE_LIBRARY");
    let request = source.request(ident.unraw().to_string(), false, options);
    let library_static = source_library(&library, request, &kernels);
    let launchers = kernels.iter().map(|kernel| kernel.launcher(&library));
    let module = module_function(&library);
    Ok(quote! {
        #(#others)*
        #[doc(hidden)]
        #[allow(dead_code)]
        mod __metal_oxide_kernels {
            use super::*;
            #(#functions)*
        }
        #values
        #library_static
        #module
        #(#launchers)*
    })
}

/// The body of a module at `location` that launches the kernels of the kernel crate in `path`.
fn kernel_crate(
    path: &LitStr,
    location: Option<(std::path::PathBuf, (usize, usize))>,
    ident: &syn::Ident,
    options: ModuleOptions,
) -> syn::Result<Tokens> {
    let item = ident.unraw().to_string();
    let Some(built) = device::kernel_crate(path, location, &item, options.math_mode)? else {
        return Ok(Tokens::new());
    };
    let mirrors = Mirrors::new(&built.abi)?;
    let records = mirrors.definitions()?;
    let library = format_ident!("__METAL_OXIDE_LIBRARY");
    let library_static = device::library(&library, Some(&built))?;
    let launchers = built
        .abi
        .kernels
        .iter()
        .map(|kernel| Ok(mirrors.kernel(kernel)?.launcher(&library)))
        .collect::<syn::Result<Vec<_>>>()?;
    let module = module_function(&library);
    Ok(quote! {
        #records
        #library_static
        #module
        #(#launchers)*
    })
}

fn is_kernel(attribute: &Attribute) -> bool {
    attribute
        .path()
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "kernel")
}
