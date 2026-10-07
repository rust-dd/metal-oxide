use rustc_hir::{Safety, def::DefKind, def_id::DefId};
use rustc_middle::ty::{Instance, Ty, TyCtxt, TypingEnv};
use rustc_span::Symbol;

pub(crate) fn block_shape(
    tcx: TyCtxt<'_>,
    definition: DefId,
) -> Result<Option<[u32; 3]>, (rustc_span::Span, String)> {
    let path = [Symbol::intern("metal_oxide"), Symbol::intern("block_shape")];
    let Some(attribute) = tcx.get_attrs_by_path(definition, &path).next() else {
        return Ok(None);
    };
    let shape = attribute
        .value_str()
        .and_then(|value| {
            value
                .as_str()
                .split(',')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()
                .ok()
        })
        .and_then(|values| <[u32; 3]>::try_from(values).ok());
    match shape {
        Some(shape)
            if shape
                .into_iter()
                .try_fold(1_u32, |n, axis| {
                    (axis != 0).then(|| n.checked_mul(axis)).flatten()
                })
                .is_some() =>
        {
            Ok(Some(shape))
        }
        _ => Err((
            tcx.def_span(definition),
            "invalid kernel block shape".into(),
        )),
    }
}

pub(crate) fn kernels<'tcx>(tcx: TyCtxt<'tcx>) -> (Vec<Instance<'tcx>>, Vec<Instance<'tcx>>) {
    let marker = [Symbol::intern("metal_oxide"), Symbol::intern("kernel")];
    let mut entries = Vec::new();
    for definition in tcx.hir_crate_items(()).definitions() {
        let definition = definition.to_def_id();
        if tcx.get_attrs_by_path(definition, &marker).next().is_some() && validate(tcx, definition)
        {
            entries.push(Instance::mono(tcx, definition));
        }
    }
    tcx.dcx().abort_if_errors();
    let instances = crate::monomorphize::reachable(tcx, &entries);
    (entries, instances)
}

fn validate(tcx: TyCtxt<'_>, definition: DefId) -> bool {
    let error = |message: &str| {
        tcx.dcx()
            .span_err(tcx.def_span(definition), message.to_owned());
        false
    };
    if tcx.def_kind(definition) != DefKind::Fn {
        return error("#[kernel] requires a free function");
    }
    if tcx.generics_of(definition).count() != 0 {
        return error("kernel entrypoints must not have generic parameters");
    }
    let signature = tcx
        .normalize_erasing_regions(
            TypingEnv::fully_monomorphized(),
            tcx.fn_sig(definition).instantiate_identity(),
        )
        .skip_binder();
    if signature.safety() != Safety::Unsafe {
        return error("kernel entrypoints must be unsafe");
    }
    if !signature.abi().is_rustic_abi() {
        return error("kernel entrypoints must use the Rust ABI");
    }
    if !signature.output().is_unit() {
        return error("kernel entrypoints must return ()");
    }
    crate::trace(|| {
        println!(
            "kernel: {} parameters={}",
            tcx.item_name(definition),
            signature.inputs().len()
        );
    });
    let mut valid = true;
    for (index, &ty) in signature.inputs().iter().enumerate() {
        match parameter(tcx, ty) {
            Some(description) => crate::trace(|| println!("parameter {index}: {description}")),
            None => {
                error(&format!("unsupported kernel parameter type: {ty}"));
                valid = false;
            }
        }
    }
    valid
}

fn parameter<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<String> {
    use metal_oxide_ir::{Access, Element, Scalar, Type};
    let (scalar, element, prefix) = match crate::types::parameter(tcx, ty)? {
        Type::Scalar(scalar) => (scalar, ty, "scalar"),
        Type::Buffer {
            element, access, ..
        } => {
            let rustc_middle::ty::Adt(_, arguments) = ty.kind() else {
                return None;
            };
            let prefix = match access {
                Access::Read => "read_buffer",
                Access::Write => "write_buffer",
                Access::Atomic => "atomic_buffer",
                Access::ReadWrite => return None,
            };
            let scalar = match element {
                Element::Scalar(scalar) => scalar,
                Element::Aggregate(_) => {
                    return Some(format!(
                        "{prefix}<{}> address_space=device",
                        arguments.type_at(0)
                    ));
                }
            };
            (scalar, arguments.type_at(0), prefix)
        }
        Type::Aggregate(_) => return Some(format!("value<{ty}>")),
        _ => return None,
    };
    let name = match scalar {
        Scalar::F32 => "f32",
        Scalar::F16 => "f16",
        Scalar::U32 => "u32",
        Scalar::I32 => "i32",
        Scalar::U8 => "u8",
        Scalar::U16 => "u16",
        Scalar::I8 => "i8",
        Scalar::I16 => "i16",
        Scalar::Bool | Scalar::Usize => return None,
    };
    let layout = tcx
        .layout_of(TypingEnv::fully_monomorphized().as_query_input(element))
        .ok()?;
    Some(if prefix == "scalar" {
        format!(
            "scalar<{name}> size={} align={}",
            layout.size.bytes(),
            layout.align.abi.bytes()
        )
    } else {
        format!(
            "{prefix}<{name}> address_space=device element_size={} element_align={}",
            layout.size.bytes(),
            layout.align.abi.bytes()
        )
    })
}
