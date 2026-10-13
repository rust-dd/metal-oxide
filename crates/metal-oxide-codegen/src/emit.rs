use crate::{
    checks::{CheckTable, FAIL},
    control::StructuredGraph,
    expressions::type_name,
    interface::KernelInterfaces,
};
use metal_oxide_artifact::MathMode;
use metal_oxide_ir::*;
use std::fmt::Write;

pub(crate) struct ModuleEmitter<'a> {
    module: &'a Module,
    interfaces: &'a KernelInterfaces,
    analyses: &'a [FunctionAnalysis],
    checks: &'a CheckTable,
    math_mode: MathMode,
    output: String,
}

impl<'a> ModuleEmitter<'a> {
    pub(crate) fn new(
        module: &'a Module,
        interfaces: &'a KernelInterfaces,
        analyses: &'a [FunctionAnalysis],
        checks: &'a CheckTable,
        math_mode: MathMode,
    ) -> Self {
        Self {
            module,
            interfaces,
            analyses,
            checks,
            math_mode,
            output: String::new(),
        }
    }

    pub(crate) fn emit(mut self) -> Result<String, Error> {
        let module = self.module;
        let simd = self.interfaces.simd;
        self.output.push_str("#include <metal_stdlib>\n");
        if self.interfaces.logging {
            self.output.push_str("#include <metal_logging>\n");
        }
        self.output.push_str("using namespace metal;\n");
        self.output.push_str(match self.math_mode {
            MathMode::Precise => "#pragma METAL fp math_mode(safe)\n#pragma STDC FP_CONTRACT OFF\n",
            MathMode::Fast => {
                "#pragma METAL fp math_mode(relaxed)\n#pragma METAL fp contract(fast)\n"
            }
        });
        self.output.push('\n');
        self.output.push_str("struct metal_oxide_context {\n    uint3 thread_idx;\n    uint3 block_idx;\n    uint3 block_dim;\n    uint3 grid_dim;\n    uint simd_lane;\n    uint simd_size;\n    uint simd_group;\n    uint simd_count;\n    const device uint* gate;\n    device atomic_uint* failure;\n    thread bool* failed;\n};\n\n");
        self.output.push_str(&crate::numeric::helpers(module)?);
        self.output.push_str(&crate::atomics::helpers(module)?);
        if module.types.pointers().next().is_some() {
            // Pointers may target aggregates, and `Option` records may hold pointers.
            for (id, _) in module.types.iter() {
                writeln!(self.output, "struct metal_oxide_aggregate_{id};").unwrap();
            }
            for (id, pointer) in module.types.pointers() {
                self.output
                    .push_str(&crate::expressions::pointer_declaration(id, pointer));
            }
            self.output.push('\n');
        }
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
        self.output.push_str(&crate::simd::helpers(module));
        self.output
            .push_str(&crate::floating::helpers(module, self.math_mode));
        if self.analyses.iter().any(|analysis| analysis.fallible) {
            self.output.push_str(FAIL);
        }
        for (id, f) in module.functions.iter().enumerate() {
            if !f.kernel {
                let signature = self.signature(f, id)?;
                writeln!(self.output, "{signature};").unwrap();
            }
        }
        self.output.push('\n');
        for (id, function) in module.functions.iter().enumerate() {
            let graph = StructuredGraph::new(function, &self.analyses[id])?;
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
            let used = crate::unused::Used::new(function, &self.analyses[id]);
            let mut body = String::new();
            crate::structured::body(
                crate::structured::Context {
                    module,
                    function,
                    id,
                    analyses: self.analyses,
                    checks: self.checks,
                    math_mode: self.math_mode,
                },
                graph,
                &used,
                &mut body,
            )?;
            let signature = self.signature(function, id)?;
            writeln!(self.output, "{signature} {{").unwrap();
            if function.kernel && body.contains("metal_oxide_ctx") {
                let simd_values = if simd {
                    ", metal_oxide_simd_lane, metal_oxide_simd_size, metal_oxide_simd_group, metal_oxide_simd_count"
                } else {
                    ", 0, 0, 0, 0"
                };
                let failure = if self.analyses[id].fallible {
                    "metal_oxide_failure"
                } else {
                    "nullptr"
                };
                self.output
                    .push_str("    bool metal_oxide_failed = false;\n");
                self.output.push_str(&format!("    metal_oxide_context metal_oxide_ctx = {{metal_oxide_thread_idx, metal_oxide_block_idx, metal_oxide_block_dim, metal_oxide_grid_dim{simd_values}, metal_oxide_gate, {failure}, &metal_oxide_failed}};\n"));
            }
            for statement in function
                .blocks
                .iter()
                .enumerate()
                .filter(|(block, _)| self.analyses[id].cfg.is_reachable(*block))
                .flat_map(|(_, b)| &b.statements)
            {
                if let Expression::ThreadgroupAlloc {
                    id,
                    element,
                    length,
                    access,
                } = statement.value
                {
                    writeln!(
                        self.output,
                        "    threadgroup {}{} metal_oxide_shared_{id}[{length}];",
                        if access == Access::Atomic {
                            "atomic_"
                        } else {
                            ""
                        },
                        type_name(Type::Scalar(element))
                    )
                    .unwrap();
                }
            }
            for (local, &ty) in function.locals.iter().enumerate() {
                let parameter = (1..=function.parameters).contains(&local);
                if matches!(ty, Type::Unit | Type::Never)
                    || (!function.kernel && parameter)
                    || (!parameter && !used.local(local))
                {
                    continue;
                }
                if function.kernel && parameter {
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
            self.output.push_str(&body);
            self.output.push_str("}\n\n");
        }
        Ok(self.output)
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
                let binding_kind = if matches!(
                    ty,
                    Type::Buffer {
                        address_space: AddressSpace::Threadgroup,
                        ..
                    }
                ) {
                    "threadgroup"
                } else {
                    "buffer"
                };
                let ty = match ty {
                    Type::Scalar(_) | Type::Aggregate(_) => format!("constant {} &", type_name(ty)),
                    _ => type_name(ty),
                };
                format!(
                    "{ty} {name} [[{binding_kind}({})]]",
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
            parameters.push(format!(
                "const device uint* metal_oxide_gate [[buffer({})]]",
                kernel.gate_binding
            ));
            if kernel.fallible {
                parameters.push(format!(
                    "device atomic_uint* metal_oxide_failure [[buffer({})]]",
                    kernel.gate_binding + 1
                ));
            }
            if kernel.lengths {
                parameters.push(format!(
                    "constant ulong* metal_oxide_lengths [[buffer({})]]",
                    kernel.gate_binding + 1 + u32::from(kernel.fallible)
                ));
            }
            Ok(format!(
                "kernel void {}({})",
                kernel.name,
                parameters.join(", ")
            ))
        } else {
            parameters.push("metal_oxide_context metal_oxide_ctx".into());
            Ok(format!(
                "static inline {} metal_oxide_fn_{id}({})",
                type_name(function.locals[0]),
                parameters.join(", ")
            ))
        }
    }
}
