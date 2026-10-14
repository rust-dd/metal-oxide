use metal_oxide_artifact::MathMode;
use proc_macro2::TokenStream;
use syn::{Expr, ExprLit, ExprTuple, Lit, LitStr, parse::Parser};

/// The arguments of `#[kernel(...)]` that shape its launch function; the device build checks
/// `requires(...)` itself.
pub(crate) struct KernelOptions {
    pub(crate) block: Option<[u32; 3]>,
}

impl KernelOptions {
    pub(crate) fn parse(tokens: TokenStream) -> syn::Result<Self> {
        let mut block = None;
        let mut requires = false;
        syn::meta::parser(|meta| {
            if meta.path.is_ident("block") && block.is_none() {
                block = Some(block_shape(&meta.value()?.parse::<ExprTuple>()?)?);
                Ok(())
            } else if meta.path.is_ident("requires") && !requires {
                requires = true;
                let relations;
                syn::parenthesized!(relations in meta.input);
                relations.parse::<TokenStream>().map(drop)
            } else {
                Err(meta.error("expected one `block = (X, Y, Z)` and one `requires(...)`"))
            }
        })
        .parse2(tokens)?;
        Ok(Self { block })
    }
}

/// The arguments of `#[kernels(...)]`.
pub(crate) struct ModuleOptions {
    pub(crate) math_mode: MathMode,
    /// Crates besides the device API that the kernels use, as named in code.
    pub(crate) dependencies: Vec<String>,
    /// The directory of a kernel crate whose kernels the module launches.
    pub(crate) path: Option<LitStr>,
}

impl ModuleOptions {
    pub(crate) fn parse(tokens: TokenStream) -> syn::Result<Self> {
        let mut math_mode = None;
        let mut dependencies = None;
        let mut path = None;
        syn::meta::parser(|meta| {
            if meta.path.is_ident("math") && math_mode.is_none() {
                let mode = meta.value()?.parse::<LitStr>()?;
                math_mode = Some(
                    mode.value()
                        .parse::<MathMode>()
                        .map_err(|error| syn::Error::new(mode.span(), error))?,
                );
                Ok(())
            } else if meta.path.is_ident("dependencies") && dependencies.is_none() {
                let mut names = Vec::new();
                meta.parse_nested_meta(|dependency| {
                    let name = dependency.path.require_ident()?;
                    names.push(name.to_string());
                    Ok(())
                })?;
                dependencies = Some(names);
                Ok(())
            } else if meta.path.is_ident("path") && path.is_none() {
                path = Some(meta.value()?.parse::<LitStr>()?);
                Ok(())
            } else {
                Err(meta.error(
                    "expected one each of `math = \"precise\" | \"fast\"`, `dependencies(...)` and `path = \"...\"`",
                ))
            }
        })
        .parse2(tokens)?;
        if path.is_some() && dependencies.is_some() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "a kernel crate names its dependencies in its own Cargo.toml",
            ));
        }
        Ok(Self {
            math_mode: math_mode.unwrap_or_default(),
            dependencies: dependencies.unwrap_or_default(),
            path,
        })
    }
}

fn block_shape(tuple: &ExprTuple) -> syn::Result<[u32; 3]> {
    let axes = tuple
        .elems
        .iter()
        .map(|axis| match axis {
            Expr::Lit(ExprLit {
                lit: Lit::Int(value),
                ..
            }) => value.base10_parse::<u32>().ok().filter(|&value| value > 0),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    match axes.as_deref() {
        Some(&[x, y, z]) if x.checked_mul(y).and_then(|xy| xy.checked_mul(z)).is_some() => {
            Ok([x, y, z])
        }
        _ => Err(syn::Error::new_spanned(
            tuple,
            "a block shape is three positive integer literals",
        )),
    }
}
