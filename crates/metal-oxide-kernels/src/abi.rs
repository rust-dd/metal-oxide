use crate::launch::{Access, Kernel, Kind};
use metal_oxide_artifact::{Abi, ParameterType};
use proc_macro2::Span;

/// Checks the launch functions read from the source against the kernels the compiler built.
pub(crate) fn check(kernels: &[Kernel], abi: &Abi, span: Span) -> syn::Result<()> {
    let mismatch = |kernel: &str, detail: String| {
        Err(syn::Error::new(
            span,
            format!("the GPU build of `{kernel}` does not match its source: {detail}"),
        ))
    };
    for kernel in kernels {
        let name = kernel.name.to_string();
        let Some(compiled) = abi.kernels.iter().find(|compiled| compiled.name == name) else {
            return mismatch(&name, "the compiler found no such kernel".into());
        };
        if compiled.parameters.len() != kernel.parameters.len() {
            return mismatch(&name, "the parameter counts differ".into());
        }
        for (parameter, built) in kernel.parameters.iter().zip(&compiled.parameters) {
            let matches = match (&parameter.kind, &built.ty) {
                (Kind::Value(_), ParameterType::Value { .. })
                | (Kind::Threadgroup(_), ParameterType::Threadgroup { .. }) => true,
                (Kind::Buffer(access, _), ParameterType::Buffer { access: built, .. }) => {
                    *access == Access::from(*built)
                }
                _ => false,
            };
            if !matches || parameter.name != built.name {
                return mismatch(
                    &name,
                    format!("parameter `{}` has another kind", parameter.name),
                );
            }
        }
        if compiled.safe != kernel.safe {
            return mismatch(&name, "the compiler disagrees on its safety".into());
        }
        if compiled.required_block != kernel.block {
            return mismatch(&name, "the block shapes differ".into());
        }
    }
    if let Some(extra) = abi
        .kernels
        .iter()
        .find(|compiled| !kernels.iter().any(|kernel| kernel.name == compiled.name))
    {
        return mismatch(&extra.name, "it has no `#[kernel]` function here".into());
    }
    Ok(())
}

impl From<metal_oxide_artifact::Access> for Access {
    fn from(access: metal_oxide_artifact::Access) -> Self {
        match access {
            metal_oxide_artifact::Access::Read => Self::Read,
            metal_oxide_artifact::Access::Write => Self::Write,
            metal_oxide_artifact::Access::ReadWrite => Self::ReadWrite,
            metal_oxide_artifact::Access::Atomic => Self::Atomic,
        }
    }
}
