use crate::{
    control::StructuredGraph,
    expressions::{aggregate_field, type_name},
    interface::KernelInterfaces,
};
use metal_oxide_ir::*;
use std::{collections::BTreeSet, fmt::Write};

pub(crate) struct ModuleEmitter<'a> {
    module: &'a Module,
    interfaces: &'a KernelInterfaces,
    output: String,
}

impl<'a> ModuleEmitter<'a> {
    pub(crate) fn new(module: &'a Module, interfaces: &'a KernelInterfaces) -> Self {
        Self {
            module,
            interfaces,
            output: String::new(),
        }
    }

    pub(crate) fn emit(mut self) -> Result<String, Error> {
        let module = self.module;
        let simd = self.interfaces.simd;
        self.output = String::from(
            "#include <metal_stdlib>\nusing namespace metal;\n#pragma STDC FP_CONTRACT OFF\n\nstruct metal_oxide_context {\n    uint3 thread_idx;\n    uint3 block_idx;\n    uint3 block_dim;\n    uint3 grid_dim;\n    uint simd_lane;\n    uint simd_size;\n    uint simd_group;\n    uint simd_count;\n};\n\n",
        );
        self.output.push_str(&crate::numeric::helpers(module)?);
        for (id, aggregate) in module.types.iter() {
            writeln!(self.output, "struct metal_oxide_aggregate_{id} {{").unwrap();
            match aggregate {
                Aggregate::Record { .. } | Aggregate::Tuple(_) => {
                    for (field, ty) in aggregate.component_types().enumerate() {
                        writeln!(self.output, "    {} f{field};", type_name(ty)).unwrap();
                    }
                }
                Aggregate::Array { element, length } => {
                    writeln!(
                        self.output,
                        "    {} elements[{length}];",
                        type_name(*element)
                    )
                    .unwrap();
                }
            }
            self.output.push_str("};\n\n");
        }
        self.aggregate_updates()?;
        for (id, f) in module.functions.iter().enumerate() {
            if !f.kernel {
                let signature = self.signature(f, id)?;
                writeln!(self.output, "{signature};").unwrap();
            }
        }
        self.output.push('\n');
        for (id, function) in module.functions.iter().enumerate() {
            let graph = StructuredGraph::new(function)?;
            for (index, block) in function.blocks.iter().enumerate() {
                if !graph.cfg.is_reachable(index) {
                    continue;
                }
                if block.terminator == Terminator::Unreachable {
                    return Err(Error::new(
                        &block.source,
                        "reachable unreachable terminator is unsupported",
                    ));
                }
            }
            let signature = self.signature(function, id)?;
            writeln!(self.output, "{signature} {{").unwrap();
            if function.kernel {
                let simd_values = if simd {
                    ", metal_oxide_simd_lane, metal_oxide_simd_size, metal_oxide_simd_group, metal_oxide_simd_count"
                } else {
                    ", 0, 0, 0, 0"
                };
                self.output.push_str(&format!("    metal_oxide_context metal_oxide_ctx = {{metal_oxide_thread_idx, metal_oxide_block_idx, metal_oxide_block_dim, metal_oxide_grid_dim{simd_values}}};\n"));
            }
            for statement in function
                .blocks
                .iter()
                .enumerate()
                .filter(|(id, _)| graph.cfg.is_reachable(*id))
                .flat_map(|(_, b)| &b.statements)
            {
                if let Expression::ThreadgroupAlloc {
                    id,
                    element,
                    length,
                } = statement.value
                {
                    writeln!(
                        self.output,
                        "    threadgroup {} metal_oxide_shared_{id}[{length}];",
                        type_name(Type::Scalar(element))
                    )
                    .unwrap();
                }
            }
            for (local, &ty) in function.locals.iter().enumerate() {
                if matches!(ty, Type::Unit | Type::Never)
                    || (!function.kernel && (1..=function.parameters).contains(&local))
                {
                    continue;
                }
                if function.kernel && (1..=function.parameters).contains(&local) {
                    writeln!(
                        self.output,
                        "    {} v{local} = metal_oxide_arg_{local};",
                        type_name(ty)
                    )
                    .unwrap();
                } else {
                    writeln!(self.output, "    {} v{local};", type_name(ty)).unwrap();
                }
            }
            crate::structured::body(module, function, graph, &mut self.output)?;
            self.output.push_str("}\n\n");
        }
        Ok(self.output)
    }

    fn aggregate_updates(&mut self) -> Result<(), Error> {
        let mut updates = BTreeSet::new();
        for function in &self.module.functions {
            for statement in function.blocks.iter().flat_map(|block| &block.statements) {
                if let Expression::AggregateUpdate {
                    aggregate, field, ..
                } = &statement.value
                {
                    let Type::Aggregate(id) =
                        operand_type(self.module, function, aggregate, &statement.source)?
                    else {
                        unreachable!("validated aggregate update")
                    };
                    updates.insert((id, *field));
                }
            }
        }
        for (id, field) in updates {
            let ty = type_name(Type::Aggregate(id));
            let component = self.module.types.get(id).unwrap().field(field).unwrap();
            writeln!(
                self.output,
                "inline {ty} metal_oxide_update_{id}_{field}({ty} value, {} replacement) {{\n    {} = replacement;\n    return value;\n}}\n",
                type_name(component),
                aggregate_field(self.module, id, "value", field),
            )
            .unwrap();
        }
        Ok(())
    }

    fn signature(&self, function: &Function, id: usize) -> Result<String, Error> {
        let simd = self.interfaces.simd;
        let kernel = self.interfaces.kernels[id].as_ref();
        if function.locals[0] == Type::Never
            || function.locals[1..=function.parameters]
                .iter()
                .any(|t| matches!(t, Type::Unit | Type::Never))
        {
            return Err(Error::new(
                &function.source,
                "unit/never parameters and diverging functions are unsupported",
            ));
        }
        let mut parameters = Vec::new();
        for local in 1..=function.parameters {
            let ty = function.locals[local];
            parameters.push(if let Some(kernel) = kernel {
                let name = format!("metal_oxide_arg_{local}");
                let ty = match ty {
                    Type::Scalar(_) | Type::Aggregate(_) => format!("constant {} &", type_name(ty)),
                    _ => type_name(ty),
                };
                format!(
                    "{ty} {name} [[buffer({})]]",
                    kernel.parameters[local - 1].binding
                )
            } else {
                format!("{} v{local}", type_name(ty))
            });
        }
        if let Some(kernel) = kernel {
            for (name, builtin) in [
                ("thread_idx", "thread_position_in_threadgroup"),
                ("block_idx", "threadgroup_position_in_grid"),
                ("block_dim", "threads_per_threadgroup"),
                ("grid_dim", "threadgroups_per_grid"),
            ] {
                parameters.push(format!("uint3 metal_oxide_{name} [[{builtin}]]"));
            }
            if simd {
                for (name, builtin) in [
                    ("lane", "thread_index_in_simdgroup"),
                    ("size", "threads_per_simdgroup"),
                    ("group", "simdgroup_index_in_threadgroup"),
                    ("count", "simdgroups_per_threadgroup"),
                ] {
                    parameters.push(format!("uint metal_oxide_simd_{name} [[{builtin}]]"));
                }
            }
            Ok(format!(
                "kernel void {}({})",
                kernel.name,
                parameters.join(", ")
            ))
        } else {
            parameters.push("metal_oxide_context metal_oxide_ctx".into());
            Ok(format!(
                "inline {} metal_oxide_fn_{id}({})",
                type_name(function.locals[0]),
                parameters.join(", ")
            ))
        }
    }
}
