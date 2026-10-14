use crate::launch::Kernel;
use proc_macro2::TokenStream;
use quote::quote;
use std::collections::BTreeSet;
use syn::{Fields, Item, ItemStruct, Type, visit::Visit};

/// `GpuValue` for the module's records that kernel parameters hold, directly or in fields, so
/// the host passes them in its own type. They must be `#[repr(C)]` with named fields.
pub(crate) fn gpu_values(items: &[Item], kernels: &[Kernel]) -> syn::Result<TokenStream> {
    let records = items
        .iter()
        .filter_map(|item| match item {
            Item::Struct(record) => Some(record),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut pending = kernels
        .iter()
        .flat_map(Kernel::named_types)
        .flat_map(names)
        .collect::<Vec<_>>();
    let mut used = BTreeSet::new();
    while let Some(name) = pending.pop() {
        if let Some(record) = records.iter().find(|record| record.ident == name)
            && used.insert(name)
        {
            pending.extend(record.fields.iter().flat_map(|field| names(&field.ty)));
        }
    }
    records
        .iter()
        .filter(|record| used.contains(&record.ident.to_string()))
        .map(|record| gpu_value(record, &record.ident.to_string()))
        .collect()
}

/// `GpuValue` for a `#[repr(C)]` record whose GPU layout is named `label`.
pub(crate) fn gpu_value(record: &ItemStruct, label: &str) -> syn::Result<TokenStream> {
    let repr_c = record.attrs.iter().any(|attribute| {
        attribute.path().is_ident("repr")
            && attribute
                .parse_args::<syn::Ident>()
                .is_ok_and(|repr| repr == "C")
    });
    let Fields::Named(fields) = &record.fields else {
        return Err(syn::Error::new_spanned(
            &record.ident,
            "a record that kernels share with the host needs named fields",
        ));
    };
    if !repr_c || !record.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &record.ident,
            "a record that kernels share with the host is a non-generic `#[repr(C)]` struct",
        ));
    }
    let name = &record.ident;
    let fields = fields
        .named
        .iter()
        .map(|field| (field.ident.as_ref().unwrap(), &field.ty))
        .collect::<Vec<_>>();
    let layouts = fields.iter().map(|(field, ty)| {
        let label = field.to_string();
        quote!((::std::string::String::from(#label), <#ty as ::metal_oxide::GpuValue>::layout()?))
    });
    let zeroed = fields
        .iter()
        .map(|(field, ty)| quote!(#field: <#ty as ::metal_oxide::GpuValue>::zeroed()));
    let range = |field, ty| quote!(::core::mem::offset_of!(Self, #field)..::core::mem::offset_of!(Self, #field) + <#ty as ::metal_oxide::GpuValue>::SIZE);
    let encode = fields.iter().map(|(field, ty)| {
        let range = range(field, ty);
        quote!(<#ty as ::metal_oxide::GpuValue>::encode(self.#field, &mut bytes[#range]);)
    });
    let decode = fields.iter().map(|(field, ty)| {
        let range = range(field, ty);
        quote!(#field: <#ty as ::metal_oxide::GpuValue>::decode(&bytes[#range]))
    });
    Ok(quote! {
        impl ::metal_oxide::GpuValue for #name {
            const SIZE: ::core::primitive::usize = ::core::mem::size_of::<Self>();
            const ALIGNMENT: ::core::primitive::usize = ::core::mem::align_of::<Self>();
            fn layout() -> ::metal_oxide::Result<::metal_oxide::Layout> {
                ::core::result::Result::Ok(::metal_oxide::Layout::record(#label, ::std::vec![#(#layouts),*])?)
            }
            fn zeroed() -> Self {
                Self { #(#zeroed),* }
            }
            fn encode(self, bytes: &mut [::core::primitive::u8]) {
                bytes.fill(0);
                #(#encode)*
            }
            fn decode(bytes: &[::core::primitive::u8]) -> Self {
                Self { #(#decode),* }
            }
        }
    })
}

/// The last segments of the paths that a type names.
fn names(ty: &Type) -> Vec<String> {
    struct Names(Vec<String>);
    impl Visit<'_> for Names {
        fn visit_type_path(&mut self, path: &syn::TypePath) {
            if let Some(last) = path.path.segments.last() {
                self.0.push(last.ident.to_string());
            }
            syn::visit::visit_type_path(self, path);
        }
    }
    let mut names = Names(Vec::new());
    names.visit_type(ty);
    names.0
}
