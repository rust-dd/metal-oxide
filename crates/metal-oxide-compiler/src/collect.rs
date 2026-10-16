use rustc_hir::{def::DefKind, def_id::DefId};
use rustc_middle::ty::{Instance, Ty, TyCtxt, TypingEnv};
use rustc_span::Symbol;

mod instances;

pub(crate) struct Entry<'tcx> {
    pub(crate) name: String,
    pub(crate) instance: Instance<'tcx>,
}

pub(crate) fn block_shape(
    tcx: TyCtxt<'_>,
    definition: DefId,
) -> Result<Option<[u32; 3]>, (rustc_span::Span, String)> {
    let path = [
        Symbol::intern("metal_oxide_compiler"),
        Symbol::intern("block_shape"),
    ];
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

/// The relations of a kernel's `requires(...)` attributes as written, with the attribute's span.
pub(crate) fn requires(
    tcx: TyCtxt<'_>,
    definition: DefId,
) -> Result<Vec<(String, rustc_span::Span)>, (rustc_span::Span, String)> {
    let path = [
        Symbol::intern("metal_oxide_compiler"),
        Symbol::intern("requires"),
    ];
    let mut relations = Vec::new();
    for attribute in tcx.get_attrs_by_path(definition, &path) {
        let rustc_attr_ir::Attribute::Unparsed(item) = attribute else {
            unreachable!("tool attributes stay unparsed")
        };
        let span = match &item.args {
            rustc_attr_ir::AttrArgs::Delimited(arguments) => arguments.dspan.entire(),
            _ => item.span,
        };
        let text = tcx
            .sess
            .source_map()
            .span_to_snippet(span)
            .ok()
            .and_then(|text| Some(text.strip_prefix('(')?.strip_suffix(')')?.to_owned()))
            .ok_or((
                span,
                "kernel requirements must be written in the source".into(),
            ))?;
        let mut depth = 0_i32;
        let mut start = 0;
        for (index, character) in text.char_indices().chain([(text.len(), ',')]) {
            match character {
                '(' => depth += 1,
                ')' => depth -= 1,
                ',' if depth == 0 => {
                    let relation = text[start..index].trim();
                    if !relation.is_empty() {
                        relations.push((relation.to_owned(), span));
                    }
                    start = index + 1;
                }
                _ => {}
            }
        }
    }
    Ok(relations)
}

/// The name a kernel parameter has in the source, from its debug information.
pub(crate) fn parameter_name(body: &rustc_middle::mir::Body<'_>, index: usize) -> Option<String> {
    let local = body.args_iter().nth(index)?;
    let name = body
        .var_debug_info
        .iter()
        .find_map(|debug| match &debug.value {
            rustc_middle::mir::VarDebugInfoContents::Place(place)
                if place.local == local && place.projection.is_empty() =>
            {
                Some(debug.name.to_string())
            }
            _ => None,
        })?;
    Some(name.strip_prefix("r#").unwrap_or(&name).to_owned())
}

/// Kernel entrypoints, after reporting unsupported code they can reach.
pub(crate) fn kernels<'tcx>(tcx: TyCtxt<'tcx>) -> Vec<Entry<'tcx>> {
    let marker = [
        Symbol::intern("metal_oxide_compiler"),
        Symbol::intern("kernel"),
    ];
    let definitions = tcx.hir_crate_items(()).definitions().collect::<Vec<_>>();
    let mut entries = instances::collect(tcx, &definitions);
    for definition in definitions {
        let definition = definition.to_def_id();
        if tcx.get_attrs_by_path(definition, &marker).next().is_none() {
            continue;
        }
        if tcx.def_kind(definition) != DefKind::Fn {
            tcx.dcx().span_err(
                tcx.def_span(definition),
                "#[kernel] requires a free function",
            );
        } else if tcx
            .generics_of(definition)
            .own_params
            .iter()
            .all(|parameter| {
                matches!(
                    parameter.kind,
                    rustc_middle::ty::GenericParamDefKind::Lifetime
                )
            })
        {
            entries.push(Entry {
                name: tcx.item_name(definition).to_string(),
                instance: Instance::mono(tcx, definition),
            });
        } else if !entries
            .iter()
            .any(|entry| entry.instance.def_id() == definition)
        {
            tcx.dcx().span_err(
                tcx.def_span(definition),
                "generic kernels require explicit instances(name(Type, CONST), ...)",
            );
        }
    }
    entries.retain(|entry| validate(tcx, entry));
    tcx.dcx().abort_if_errors();
    let instances = entries
        .iter()
        .map(|entry| entry.instance)
        .collect::<Vec<_>>();
    crate::monomorphize::validate(tcx, &instances);
    entries
}

fn validate<'tcx>(tcx: TyCtxt<'tcx>, entry: &Entry<'tcx>) -> bool {
    let definition = entry.instance.def_id();
    let error = |message: &str| {
        tcx.dcx()
            .span_err(tcx.def_span(definition), message.to_owned());
        false
    };
    if tcx.def_kind(definition) != DefKind::Fn {
        return error("#[kernel] requires a free function");
    }
    let signature = tcx
        .normalize_erasing_regions(
            TypingEnv::fully_monomorphized(),
            tcx.fn_sig(definition).instantiate(tcx, entry.instance.args),
        )
        .skip_binder();
    if !signature.abi().is_rustic_abi() {
        return error("kernel entrypoints must use the Rust ABI");
    }
    if !signature.output().is_unit() {
        return error("kernel entrypoints must return ()");
    }
    crate::trace(|| {
        println!(
            "kernel: {} parameters={}",
            entry.name,
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
    use metal_oxide_ir::{Access, AddressSpace, Element, Type};
    let (scalar, element, prefix, space) = match crate::types::parameter(tcx, ty)? {
        Type::Scalar(scalar) => (scalar, ty, "scalar", "constant"),
        Type::Buffer {
            element,
            access,
            address_space,
        } => {
            let rustc_middle::ty::Adt(_, arguments) = ty.kind() else {
                return None;
            };
            let (prefix, space) = if address_space == AddressSpace::Threadgroup {
                ("dynamic_threadgroup_buffer", "threadgroup")
            } else {
                (
                    match access {
                        Access::Read => "read_buffer",
                        Access::Write => "write_buffer",
                        Access::Atomic => "atomic_buffer",
                        Access::ReadWrite => "read_write_buffer",
                    },
                    "device",
                )
            };
            let scalar = match element {
                Element::Scalar(scalar) => scalar,
                Element::Aggregate(_) => {
                    return Some(format!(
                        "{prefix}<{}> address_space={space}",
                        arguments.type_at(0)
                    ));
                }
            };
            (scalar, arguments.type_at(0), prefix, space)
        }
        Type::Aggregate(_) => return Some(format!("value<{ty}>")),
        _ => return None,
    };
    let name = scalar.name();
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
            "{prefix}<{name}> address_space={space} element_size={} element_align={}",
            layout.size.bytes(),
            layout.align.abi.bytes()
        )
    })
}
