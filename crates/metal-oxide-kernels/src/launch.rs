use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, FnArg, GenericArgument, Ident, ItemFn, Pat, PathArguments, ReturnType, Safety, Type,
    Visibility,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Access {
    Read,
    Write,
    ReadWrite,
    Atomic,
}

pub(crate) enum Kind {
    Value(Type),
    Buffer(Access, Type),
    Threadgroup(Type),
}

pub(crate) struct Parameter {
    pub(crate) name: Ident,
    pub(crate) kind: Kind,
}

/// A kernel's launch-relevant signature, read from its source.
pub(crate) struct Kernel {
    pub(crate) name: Ident,
    visibility: Visibility,
    pub(crate) safe: bool,
    pub(crate) block: Option<[u32; 3]>,
    pub(crate) parameters: Vec<Parameter>,
    docs: Vec<Attribute>,
}

impl Kernel {
    pub(crate) fn new(function: &ItemFn, block: Option<[u32; 3]>) -> syn::Result<Self> {
        let signature = &function.sig;
        let unsupported = if !signature.generics.params.is_empty() {
            Some("kernels cannot be generic")
        } else if signature.asyncness.is_some() || signature.constness.is_some() {
            Some("kernels cannot be `async` or `const`")
        } else if signature.abi.is_some() || signature.variadic.is_some() {
            Some("kernels use the Rust ABI without variadic parameters")
        } else if !matches!(signature.output, ReturnType::Default) {
            Some("kernels return nothing")
        } else {
            None
        };
        if let Some(message) = unsupported {
            return Err(syn::Error::new_spanned(signature, message));
        }
        Ok(Self {
            name: signature.ident.clone(),
            visibility: function.vis.clone(),
            safe: matches!(signature.safety, Safety::Default),
            block,
            parameters: signature
                .inputs
                .iter()
                .map(parameter)
                .collect::<syn::Result<_>>()?,
            docs: function
                .attrs
                .iter()
                .filter(|attribute| attribute.path().is_ident("doc"))
                .cloned()
                .collect(),
        })
    }

    /// A public kernel of a kernel crate, from its ABI.
    pub(crate) fn exported(
        name: Ident,
        safe: bool,
        block: Option<[u32; 3]>,
        parameters: Vec<Parameter>,
    ) -> Self {
        Self {
            name,
            visibility: syn::parse_quote!(pub),
            safe,
            block,
            parameters,
            docs: Vec::new(),
        }
    }

    /// The type names that the parameters use, which `#[kernels]` modules may define.
    pub(crate) fn named_types(&self) -> impl Iterator<Item = &Type> {
        self.parameters
            .iter()
            .map(|parameter| match &parameter.kind {
                Kind::Value(ty) | Kind::Buffer(_, ty) | Kind::Threadgroup(ty) => ty,
            })
    }

    /// A function with the kernel's name that returns its launch from `library`.
    pub(crate) fn launcher(&self, library: &Ident) -> TokenStream {
        let Self {
            name,
            visibility,
            docs,
            ..
        } = self;
        let entry = name.to_string();
        let mut generics = Vec::new();
        let mut bounds = Vec::new();
        let mut parameters = Vec::new();
        let mut types = Vec::new();
        let mut names = Vec::new();
        let mut arguments = Vec::new();
        for (index, parameter) in self.parameters.iter().enumerate() {
            let reserved = ["device", "config", "_"].contains(&parameter.name.to_string().as_str());
            let name = if reserved || names.contains(&parameter.name) {
                format_ident!("argument_{index}")
            } else {
                parameter.name.clone()
            };
            let (ty, argument) = match &parameter.kind {
                Kind::Value(ty) => (quote!(#ty), quote!(::metal_oxide::Argument::value(*#name)?)),
                Kind::Threadgroup(element) => (
                    quote!(::core::primitive::usize),
                    quote!(::metal_oxide::Argument::threadgroup::<#element>(*#name)?),
                ),
                Kind::Buffer(access, element) => {
                    let generic = format_ident!("MetalOxideBuffer{index}");
                    let (form, view, method) = match access {
                        Access::Read => (quote!(ReadArgument), quote!(buffer), quote!(read)),
                        Access::Write => (quote!(WriteArgument), quote!(buffer_mut), quote!(write)),
                        Access::ReadWrite => (
                            quote!(WriteArgument),
                            quote!(buffer_mut),
                            quote!(read_write),
                        ),
                        Access::Atomic => {
                            (quote!(WriteArgument), quote!(buffer_mut), quote!(atomic))
                        }
                    };
                    bounds.push(quote!(#generic: ::metal_oxide::#form<#element>));
                    generics.push(generic.clone());
                    (
                        quote!(#generic),
                        quote!(::metal_oxide::Argument::#method(::metal_oxide::#form::#view(#name))),
                    )
                }
            };
            parameters.push(quote!(#name: #ty));
            types.push(ty);
            names.push(name);
            arguments.push(argument);
        }
        let count = arguments.len();
        let config = match self.block {
            Some([x, y, z]) => quote!(::metal_oxide::LaunchConfig<#x, #y, #z>),
            None => quote!(impl ::core::convert::Into<::metal_oxide::DynamicLaunchConfig>),
        };
        let documented = docs
            .iter()
            .any(|doc| quote!(#doc).to_string().contains("# Safety"));
        let (qualifier, safety) = if self.safe {
            (quote!(), quote!())
        } else if documented {
            (quote!(unsafe), quote!())
        } else {
            (
                quote!(unsafe),
                quote!(
                    #[doc = ""]
                    #[doc = "# Safety"]
                    #[doc = "The caller must satisfy the kernel's bounds, race and synchronization contracts."]
                ),
            )
        };
        quote! {
            #(#docs)*
            #safety
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            #[allow(clippy::too_many_arguments, clippy::type_complexity)]
            #visibility #qualifier fn #name<#(#generics),*>(
                device: &::metal_oxide::Device,
                config: #config,
                #(#parameters),*
            ) -> ::metal_oxide::Launch<(#(#types,)*), #count>
            where
                #(#bounds,)*
            {
                // SAFETY: the launch checks a safe kernel's parameter types and `requires`
                // relations; an unsafe kernel's contract is this function's.
                unsafe {
                    ::metal_oxide::Launch::kernel(
                        device,
                        &#library,
                        #entry,
                        config,
                        (#(#names,)*),
                        |(#(#names,)*)| ::core::result::Result::Ok([#(#arguments),*]),
                    )
                }
            }
        }
    }
}

fn parameter(input: &FnArg) -> syn::Result<Parameter> {
    let FnArg::Typed(typed) = input else {
        return Err(syn::Error::new_spanned(input, "kernels take no `self`"));
    };
    let Pat::Ident(pattern) = &*typed.pat else {
        return Err(syn::Error::new_spanned(
            &typed.pat,
            "kernel parameters are plain names",
        ));
    };
    Ok(Parameter {
        name: pattern.ident.clone(),
        kind: kind(&typed.ty)?,
    })
}

/// Device buffer parameters by their type's last segment; every other type is a value.
fn kind(ty: &Type) -> syn::Result<Kind> {
    let Type::Path(path) = ty else {
        return Ok(Kind::Value(ty.clone()));
    };
    let Some(last) = path.path.segments.last().filter(|_| path.qself.is_none()) else {
        return Ok(Kind::Value(ty.clone()));
    };
    let access = match last.ident.to_string().as_str() {
        "ReadBuffer" => Some(Access::Read),
        "WriteBuffer" | "DisjointBuffer" => Some(Access::Write),
        "ReadWriteBuffer" => Some(Access::ReadWrite),
        "AtomicBuffer" => Some(Access::Atomic),
        "DynamicThreadgroupBuffer" => None,
        _ => return Ok(Kind::Value(ty.clone())),
    };
    let element = match &last.arguments {
        PathArguments::AngleBracketed(arguments) if arguments.args.len() == 1 => {
            match &arguments.args[0] {
                GenericArgument::Type(element) => element.clone(),
                other => return Err(syn::Error::new_spanned(other, "expected an element type")),
            }
        }
        _ => return Err(syn::Error::new_spanned(last, "expected one element type")),
    };
    Ok(match access {
        Some(access) => Kind::Buffer(access, element),
        None => Kind::Threadgroup(element),
    })
}

#[cfg(test)]
mod tests;
