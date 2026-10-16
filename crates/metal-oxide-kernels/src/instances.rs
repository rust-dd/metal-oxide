use std::collections::{HashMap, HashSet};

use quote::{ToTokens, quote};
use syn::{
    AngleBracketedGenericArguments, Expr, GenericArgument, GenericParam, Ident, ItemFn, Token,
    Type,
    ext::IdentExt,
    fold::{self, Fold},
    parse::ParseStream,
};

use crate::{launch::Kernel, options::KernelOptions};

pub(crate) struct Instance {
    name: Ident,
    arguments: AngleBracketedGenericArguments,
}

pub(crate) fn parse(input: ParseStream<'_>) -> syn::Result<Vec<Instance>> {
    let mut entries = Vec::new();
    let mut names = HashSet::new();
    while !input.is_empty() {
        let name = input.parse::<Ident>()?;
        if !names.insert(name.unraw().to_string()) {
            return Err(syn::Error::new_spanned(
                name,
                "duplicate kernel instance name",
            ));
        }
        let arguments;
        syn::parenthesized!(arguments in input);
        let arguments = arguments.parse::<proc_macro2::TokenStream>()?;
        let arguments = syn::parse2::<AngleBracketedGenericArguments>(quote!(<#arguments>))?;
        entries.push(Instance { name, arguments });
        if input.is_empty() {
            break;
        }
        input.parse::<Token![,]>()?;
    }
    if entries.is_empty() {
        return Err(input.error("instances(...) needs at least one named instance"));
    }
    Ok(entries)
}

pub(crate) fn kernels(function: &ItemFn, options: KernelOptions) -> syn::Result<Vec<Kernel>> {
    if options.instances.is_empty() {
        let concrete = erase_lifetimes(function);
        return Kernel::new(&concrete, options.block).map(|kernel| vec![kernel]);
    }
    let parameters = &function.sig.generics.params;
    if parameters
        .iter()
        .all(|parameter| matches!(parameter, GenericParam::Lifetime(_)))
    {
        return Err(syn::Error::new_spanned(
            &function.sig,
            "instances(...) requires a generic kernel",
        ));
    }
    options
        .instances
        .into_iter()
        .map(|instance| Kernel::new(&specialize(function, instance)?, options.block))
        .collect()
}

fn specialize(function: &ItemFn, instance: Instance) -> syn::Result<ItemFn> {
    let function = erase_lifetimes(function);
    let parameters = &function.sig.generics.params;
    if parameters.len() != instance.arguments.args.len() {
        return Err(syn::Error::new_spanned(
            instance.arguments,
            "an instance supplies every type and const parameter",
        ));
    }
    let mut substitution = Substitution::default();
    for (parameter, argument) in parameters.iter().zip(instance.arguments.args) {
        match (parameter, argument) {
            (GenericParam::Type(parameter), GenericArgument::Type(ty)) => {
                substitution.types.insert(parameter.ident.to_string(), ty);
            }
            (GenericParam::Const(parameter), argument) => {
                let value = syn::parse2::<Expr>(argument.into_token_stream())?;
                substitution
                    .constants
                    .insert(parameter.ident.to_string(), value);
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    parameter,
                    "kernel instances take type and const arguments, without lifetime parameters",
                ));
            }
        }
    }
    let mut concrete = function;
    concrete.sig.inputs = concrete
        .sig
        .inputs
        .into_iter()
        .map(|argument| substitution.fold_fn_arg(argument))
        .collect();
    concrete.sig.generics = syn::Generics::default();
    concrete.sig.ident = instance.name;
    Ok(concrete)
}

fn erase_lifetimes(function: &ItemFn) -> ItemFn {
    struct Erase(HashSet<String>);
    impl Fold for Erase {
        fn fold_lifetime(&mut self, lifetime: syn::Lifetime) -> syn::Lifetime {
            if self.0.contains(&lifetime.ident.to_string()) {
                syn::Lifetime::new("'static", lifetime.apostrophe)
            } else {
                lifetime
            }
        }
    }
    let names = function
        .sig
        .generics
        .lifetimes()
        .map(|parameter| parameter.lifetime.ident.to_string())
        .collect::<HashSet<_>>();
    let mut function = function.clone();
    let mut erase = Erase(names);
    function.sig.inputs = function
        .sig
        .inputs
        .into_iter()
        .map(|argument| erase.fold_fn_arg(argument))
        .collect();
    function.sig.generics.params = function
        .sig
        .generics
        .params
        .into_iter()
        .filter(|parameter| !matches!(parameter, GenericParam::Lifetime(_)))
        .collect();
    function
}

#[derive(Default)]
struct Substitution {
    types: HashMap<String, Type>,
    constants: HashMap<String, Expr>,
}

impl Fold for Substitution {
    fn fold_type(&mut self, ty: Type) -> Type {
        if let Type::Path(path) = &ty
            && path.qself.is_none()
            && let Some(name) = path.path.get_ident()
            && let Some(actual) = self.types.get(&name.to_string())
        {
            return actual.clone();
        }
        fold::fold_type(self, ty)
    }

    fn fold_expr(&mut self, expr: Expr) -> Expr {
        if let Expr::Path(path) = &expr
            && path.qself.is_none()
            && let Some(name) = path.path.get_ident()
            && let Some(actual) = self.constants.get(&name.to_string())
        {
            return actual.clone();
        }
        fold::fold_expr(self, expr)
    }

    fn fold_generic_argument(&mut self, argument: GenericArgument) -> GenericArgument {
        // A bare const argument is syntactically a type until its parameter is resolved.
        if let GenericArgument::Type(Type::Path(path)) = &argument
            && path.qself.is_none()
            && let Some(name) = path.path.get_ident()
            && let Some(actual) = self.constants.get(&name.to_string())
        {
            return GenericArgument::Const(syn::parse_quote!({ #actual }));
        }
        fold::fold_generic_argument(self, argument)
    }
}
