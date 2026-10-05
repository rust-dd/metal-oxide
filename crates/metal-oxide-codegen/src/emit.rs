use crate::{
    control::Graph,
    expressions::{expression, operand, type_name},
};
use metal_oxide_ir::*;
use std::{collections::HashSet, fmt::Write};

pub(crate) fn module(module: &Module) -> Result<String, Error> {
    let mut output = String::from(
        "#include <metal_stdlib>\nusing namespace metal;\n#pragma STDC FP_CONTRACT OFF\n\nstruct metal_oxide_context {\n    uint3 thread_idx;\n    uint3 block_idx;\n    uint3 block_dim;\n    uint3 grid_dim;\n};\n\n",
    );
    output.push_str(&crate::numeric::helpers(module)?);
    for (id, f) in module.functions.iter().enumerate() {
        if !f.kernel {
            writeln!(output, "{};", signature(f, id)?).unwrap();
        }
    }
    output.push('\n');
    for (id, function) in module.functions.iter().enumerate() {
        let graph = Graph::new(function)?;
        for (index, block) in function.blocks.iter().enumerate() {
            if !graph.reachable[index] {
                continue;
            }
            match &block.terminator {
                Terminator::Assert {
                    enabled: true,
                    message,
                    ..
                } => {
                    return Err(Error::new(
                        &block.source,
                        format!("MIR assertion cannot be lowered to Metal yet: {message}"),
                    ));
                }
                Terminator::Unreachable => {
                    return Err(Error::new(
                        &block.source,
                        "reachable unreachable terminator is unsupported",
                    ));
                }
                _ => {}
            }
        }
        writeln!(output, "{} {{", signature(function, id)?).unwrap();
        if function.kernel {
            output.push_str("    metal_oxide_context metal_oxide_ctx = {metal_oxide_thread_idx, metal_oxide_block_idx, metal_oxide_block_dim, metal_oxide_grid_dim};\n");
        }
        for (local, &ty) in function.locals.iter().enumerate() {
            if matches!(ty, Type::Unit | Type::Never)
                || (!function.kernel && (1..=function.parameters).contains(&local))
            {
                continue;
            }
            if function.kernel && (1..=function.parameters).contains(&local) {
                writeln!(
                    output,
                    "    {} v{local} = metal_oxide_arg_{local};",
                    type_name(ty)
                )
                .unwrap();
            } else {
                writeln!(output, "    {} v{local};", type_name(ty)).unwrap();
            }
        }
        let mut emitter = Emitter {
            module,
            function,
            graph,
            output: &mut output,
            visited: HashSet::new(),
            indent: 1,
        };
        emitter.path(0, function.blocks.len(), &[], None)?;
        output.push_str("}\n\n");
    }
    Ok(output)
}

fn signature(function: &Function, id: usize) -> Result<String, Error> {
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
        parameters.push(if function.kernel {
            let name = format!("metal_oxide_arg_{local}");
            let ty = match ty {
                Type::Scalar(s) => format!("constant {} &", type_name(Type::Scalar(s))),
                _ => type_name(ty).into(),
            };
            format!("{ty} {name} [[buffer({})]]", local - 1)
        } else {
            format!("{} v{local}", type_name(ty))
        });
    }
    if function.kernel {
        validate_name(function)?;
        for (name, builtin) in [
            ("thread_idx", "thread_position_in_threadgroup"),
            ("block_idx", "threadgroup_position_in_grid"),
            ("block_dim", "threads_per_threadgroup"),
            ("grid_dim", "threadgroups_per_grid"),
        ] {
            parameters.push(format!("uint3 metal_oxide_{name} [[{builtin}]]"));
        }
        Ok(format!(
            "kernel void {}({})",
            function.name,
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

fn validate_name(function: &Function) -> Result<(), Error> {
    let name = &function.name;
    let valid = !name.is_empty()
        && name
            .bytes()
            .enumerate()
            .all(|(i, b)| b.is_ascii_alphabetic() || b == b'_' || (i > 0 && b.is_ascii_digit()));
    let reserved = [
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "auto",
        "bitand",
        "bitor",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "char16_t",
        "char32_t",
        "class",
        "compl",
        "const",
        "const_cast",
        "constant",
        "constexpr",
        "continue",
        "decltype",
        "default",
        "delete",
        "device",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "float",
        "for",
        "fragment",
        "friend",
        "goto",
        "half",
        "if",
        "inline",
        "int",
        "kernel",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "private",
        "public",
        "register",
        "reinterpret_cast",
        "restrict",
        "return",
        "sampler",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "template",
        "this",
        "thread",
        "thread_local",
        "threadgroup",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "uint",
        "union",
        "unsigned",
        "using",
        "vertex",
        "virtual",
        "void",
        "volatile",
        "wchar_t",
        "while",
        "xor",
        "xor_eq",
    ];
    if !valid
        || name.starts_with("metal_oxide_")
        || name.contains("__")
        || (name.starts_with('_') && name.as_bytes().get(1).is_some_and(u8::is_ascii_uppercase))
        || reserved.contains(&name.as_str())
    {
        Err(Error::new(
            &function.source,
            "kernel name is not a supported MSL identifier",
        ))
    } else {
        Ok(())
    }
}

struct Emitter<'a> {
    module: &'a Module,
    function: &'a Function,
    graph: Graph,
    output: &'a mut String,
    visited: HashSet<usize>,
    indent: usize,
}

impl Emitter<'_> {
    fn line(&mut self, value: &str) {
        writeln!(self.output, "{}{value}", "    ".repeat(self.indent)).unwrap();
    }

    fn path(
        &mut self,
        mut current: usize,
        stop: usize,
        contexts: &[usize],
        enter_loop: Option<usize>,
    ) -> Result<(), Error> {
        let exit = self.function.blocks.len();
        while current != stop && current != exit {
            if let Some(&header) = contexts.last() {
                let loop_ = &self.graph.loops[&header];
                if current == header && enter_loop != Some(header) {
                    self.line("continue;");
                    return Ok(());
                }
                if !loop_.members.contains(&current) {
                    if current == loop_.exit {
                        self.line("break;");
                        return Ok(());
                    }
                    return Err(Error::new(
                        &self.function.blocks[current].source,
                        "unsupported control flow leaving a loop",
                    ));
                }
            }
            if self.graph.loops.contains_key(&current) && enter_loop != Some(current) {
                let loop_exit = self.graph.loops[&current].exit;
                self.line("while (true) {");
                self.indent += 1;
                let mut inner = contexts.to_vec();
                inner.push(current);
                self.path(current, exit, &inner, Some(current))?;
                self.indent -= 1;
                self.line("}");
                current = loop_exit;
                continue;
            }
            let block = &self.function.blocks[current];
            if !self.visited.insert(current) {
                return Err(Error::new(
                    &block.source,
                    "control flow cannot be structured without duplicating a block",
                ));
            }
            for statement in &block.statements {
                let value = expression(
                    self.module,
                    self.function,
                    &statement.value,
                    &statement.source,
                )?;
                if self.function.locals[statement.destination] == Type::Unit {
                    if !value.is_empty() {
                        self.line(&format!("{value};"));
                    }
                } else {
                    self.line(&format!("v{} = {value};", statement.destination));
                }
            }
            match &block.terminator {
                Terminator::Goto(target)
                | Terminator::Assert {
                    enabled: false,
                    target,
                    ..
                } => current = *target,
                Terminator::Return => {
                    if self.function.locals[0] == Type::Unit {
                        self.line("return;");
                    } else {
                        self.line("return v0;");
                    }
                    return Ok(());
                }
                Terminator::Branch {
                    condition,
                    then_block,
                    else_block,
                } => {
                    let mut join = self.graph.join(current);
                    if let Some(header) = contexts.last()
                        && !self.graph.loops[header].members.contains(&join)
                    {
                        join = exit;
                    }
                    self.line(&format!("if ({}) {{", operand(self.function, condition)));
                    self.indent += 1;
                    self.path(*then_block, join, contexts, None)?;
                    self.indent -= 1;
                    self.line("} else {");
                    self.indent += 1;
                    self.path(*else_block, join, contexts, None)?;
                    self.indent -= 1;
                    self.line("}");
                    current = join;
                }
                _ => return Err(Error::new(&block.source, "unsupported terminator")),
            }
        }
        Ok(())
    }
}
