use crate::{
    launch::{Kernel, Kind, Parameter},
    values,
};
use metal_oxide_artifact::{Abi, Layout, LayoutKind, ParameterType, Scalar};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Ident, ItemStruct};

/// Host structs for the records that a kernel crate's kernels pass, one per distinct layout,
/// named after the record.
pub(crate) struct Mirrors(Vec<(Layout, Ident)>);

impl Mirrors {
    pub(crate) fn new(abi: &Abi) -> syn::Result<Self> {
        let mut mirrors = Self(Vec::new());
        for parameter in abi.kernels.iter().flat_map(|kernel| &kernel.parameters) {
            mirrors.collect(match &parameter.ty {
                ParameterType::Value { layout } => layout,
                ParameterType::Buffer { element, .. }
                | ParameterType::Threadgroup { element, .. } => element,
            })?;
        }
        Ok(mirrors)
    }

    fn collect(&mut self, layout: &Layout) -> syn::Result<()> {
        match &layout.kind {
            LayoutKind::Scalar { .. } => {}
            LayoutKind::Array { element, .. } => self.collect(element)?,
            LayoutKind::Tuple { fields } | LayoutKind::Record { fields, .. } => {
                for field in fields {
                    self.collect(&field.layout)?;
                }
            }
        }
        if let LayoutKind::Record { name, .. } = &layout.kind
            && !self.0.iter().any(|(other, _)| other == layout)
        {
            // Launch functions name their buffer generics `MetalOxideBuffer<N>`.
            let generic = |name: &str| {
                name.strip_prefix("MetalOxideBuffer")
                    .is_some_and(|index| index.parse::<u32>().is_ok())
            };
            let base = if generic(name) {
                format!("{name}Record")
            } else {
                name.clone()
            };
            let mut candidate = base.clone();
            let mut suffix = 2;
            while self.0.iter().any(|(_, ident)| *ident == candidate) {
                candidate = format!("{base}{suffix}");
                suffix += 1;
            }
            self.0.push((layout.clone(), ident(&candidate)?));
        }
        Ok(())
    }

    fn ty(&self, layout: &Layout) -> TokenStream {
        match &layout.kind {
            LayoutKind::Scalar { scalar } => scalar_type(*scalar),
            LayoutKind::Array {
                element, length, ..
            } => {
                let element = self.ty(element);
                let length = *length as usize;
                quote!([#element; #length])
            }
            LayoutKind::Tuple { fields } => {
                let fields = fields.iter().map(|field| self.ty(&field.layout));
                quote!((#(#fields,)*))
            }
            LayoutKind::Record { .. } => {
                let ident = &self.0.iter().find(|(other, _)| other == layout).unwrap().1;
                quote!(#ident)
            }
        }
    }

    /// The structs with `GpuValue` and compile-time checks that their layouts match the ABI.
    pub(crate) fn definitions(&self) -> syn::Result<TokenStream> {
        self.0
            .iter()
            .map(|(layout, record_name)| {
                let LayoutKind::Record { name, fields } = &layout.kind else {
                    unreachable!("mirrors hold records")
                };
                let names = fields
                    .iter()
                    .map(|field| ident(&field.name))
                    .collect::<syn::Result<Vec<_>>>()?;
                let types = fields.iter().map(|field| self.ty(&field.layout));
                let record = syn::parse2::<ItemStruct>(quote! {
                    #[repr(C)]
                    #[derive(Clone, Copy, Debug, PartialEq)]
                    pub struct #record_name { #(pub #names: #types),* }
                })?;
                let value = values::gpu_value(&record, name)?;
                let size = layout.size as usize;
                let alignment = layout.alignment as usize;
                let offsets = fields.iter().map(|field| field.offset as usize);
                Ok(quote! {
                    #record
                    #value
                    const _: () = {
                        assert!(::core::mem::size_of::<#record_name>() == #size);
                        assert!(::core::mem::align_of::<#record_name>() == #alignment);
                        #(assert!(::core::mem::offset_of!(#record_name, #names) == #offsets);)*
                    };
                })
            })
            .collect()
    }

    /// The launch-relevant signature of a compiled kernel.
    pub(crate) fn kernel(&self, kernel: &metal_oxide_artifact::Kernel) -> syn::Result<Kernel> {
        let parameters = kernel
            .parameters
            .iter()
            .map(|parameter| {
                let ty = |layout| syn::parse2(self.ty(layout));
                let kind = match &parameter.ty {
                    ParameterType::Value { layout } => Kind::Value(ty(layout)?),
                    ParameterType::Buffer {
                        element, access, ..
                    } => Kind::Buffer(*access, ty(element)?),
                    ParameterType::Threadgroup { element, .. } => Kind::Threadgroup(ty(element)?),
                };
                Ok(Parameter {
                    name: ident(&parameter.name).unwrap_or_else(|_| format_ident!("argument")),
                    kind,
                })
            })
            .collect::<syn::Result<_>>()?;
        Ok(Kernel::exported(
            ident(&kernel.name)?,
            kernel.safe,
            kernel.required_block,
            parameters,
        ))
    }
}

fn scalar_type(scalar: Scalar) -> TokenStream {
    match scalar {
        Scalar::F32 => quote!(::core::primitive::f32),
        Scalar::F16 => quote!(::metal_oxide::F16),
        Scalar::BF16 => quote!(::metal_oxide::BF16),
        Scalar::U32 => quote!(::core::primitive::u32),
        Scalar::U64 => quote!(::core::primitive::u64),
        Scalar::I32 => quote!(::core::primitive::i32),
        Scalar::U8 => quote!(::core::primitive::u8),
        Scalar::U16 => quote!(::core::primitive::u16),
        Scalar::I8 => quote!(::core::primitive::i8),
        Scalar::I16 => quote!(::core::primitive::i16),
    }
}

/// `name` as an identifier, raw when it is a keyword.
fn ident(name: &str) -> syn::Result<Ident> {
    syn::parse_str::<Ident>(name)
        .or_else(|_| syn::parse_str::<Ident>(&format!("r#{name}")))
        .map_err(|_| {
            syn::Error::new(
                Span::call_site(),
                format!("`{name}` from the kernel crate is not a Rust identifier"),
            )
        })
}

#[cfg(test)]
mod tests;
