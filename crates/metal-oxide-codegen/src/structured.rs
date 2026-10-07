use crate::{
    control::StructuredGraph,
    expressions::{expression, operand},
};
use metal_oxide_ir::*;
use std::{collections::HashMap, fmt::Write};

pub(crate) fn body(
    module: &Module,
    function: &Function,
    graph: StructuredGraph,
    output: &mut String,
) -> Result<(), Error> {
    Emitter {
        module,
        function,
        graph,
        output,
        visits: HashMap::new(),
        indent: 1,
    }
    .path(0, function.blocks.len(), &[], None)
}

struct Emitter<'a> {
    module: &'a Module,
    function: &'a Function,
    graph: StructuredGraph,
    output: &'a mut String,
    visits: HashMap<usize, usize>,
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
        mut enter_loop: Option<usize>,
    ) -> Result<(), Error> {
        let exit = self.function.blocks.len();
        while current != stop && current != exit {
            let enter_loop = enter_loop.take();
            if let Some(&header) = contexts.last() {
                let loop_ = &self.graph.loops[&header];
                if current == header && enter_loop != Some(header) {
                    self.line("continue;");
                    return Ok(());
                }
                if !loop_.members.contains(&current) {
                    if loop_.exits.contains(&current) {
                        if loop_.exits.len() > 1 {
                            self.line(&format!("metal_oxide_exit_{header} = {current}u;"));
                        }
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
                current = self.loop_body(current, contexts, stop)?;
                continue;
            }
            let block = &self.function.blocks[current];
            let visits = self.visits.entry(current).or_default();
            *visits += 1;
            if *visits > 8
                || (*visits > 1
                    && block.statements.iter().any(|s| {
                        matches!(
                            s.value,
                            Expression::ThreadgroupAlloc { .. }
                                | Expression::ThreadgroupBarrier
                                | Expression::SimdSum(_)
                                | Expression::SimdShuffle { .. }
                                | Expression::Call { .. }
                        )
                    }))
            {
                return Err(Error::new(
                    &block.source,
                    "control flow requires excessive or cooperative block duplication",
                ));
            }
            for statement in &block.statements {
                let value = expression(
                    self.module,
                    self.function,
                    &statement.value,
                    &statement.source,
                )?;
                if self.function.locals[statement.destination.local] == Type::Unit {
                    if !value.is_empty() {
                        self.line(&format!("{value};"));
                    }
                } else {
                    self.line(&format!(
                        "{} = {value};",
                        crate::expressions::place(
                            self.module,
                            self.function,
                            &statement.destination
                        )
                    ));
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
                    let join = self.branch_join(current, contexts, stop);
                    self.branches(
                        &[(operand(self.module, self.function, condition), *then_block)],
                        *else_block,
                        join,
                        contexts,
                    )?;
                    current = join;
                }
                Terminator::Switch {
                    discriminant,
                    cases,
                    otherwise,
                } => {
                    let join = self.branch_join(current, contexts, stop);
                    let mut groups = Vec::<(usize, Vec<Constant>)>::new();
                    for &(value, target) in cases {
                        if target == *otherwise {
                            continue;
                        }
                        if let Some((_, values)) = groups.iter_mut().find(|(id, _)| *id == target) {
                            values.push(value);
                        } else {
                            groups.push((target, vec![value]));
                        }
                    }
                    let branches = groups
                        .into_iter()
                        .map(|(target, values)| {
                            let condition = values
                                .iter()
                                .map(|value| {
                                    format!(
                                        "{} == {}",
                                        operand(self.module, self.function, discriminant),
                                        operand(
                                            self.module,
                                            self.function,
                                            &Operand::Constant(*value)
                                        ),
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join(" || ");
                            (condition, target)
                        })
                        .collect::<Vec<_>>();
                    // If-chains keep break and continue bound to their enclosing loop.
                    self.branches(&branches, *otherwise, join, contexts)?;
                    current = join;
                }
                _ => return Err(Error::new(&block.source, "unsupported terminator")),
            }
        }
        Ok(())
    }

    fn branch_join(&self, block: usize, contexts: &[usize], stop: usize) -> usize {
        if let Some(&header) = contexts.last() {
            self.graph.loop_join(
                &self.function.blocks[block].terminator.successors(),
                header,
                stop,
            )
        } else {
            self.graph.postdominators.immediate(block)
        }
    }

    fn branches(
        &mut self,
        branches: &[(String, usize)],
        otherwise: usize,
        join: usize,
        contexts: &[usize],
    ) -> Result<(), Error> {
        for (index, (condition, target)) in branches.iter().enumerate() {
            let prefix = if index == 0 { "if" } else { "else if" };
            self.line(&format!("{prefix} ({condition}) {{"));
            self.indent += 1;
            self.path(*target, join, contexts, None)?;
            self.indent -= 1;
            self.line("}");
        }
        if !branches.is_empty() {
            self.line("else {");
            self.indent += 1;
        }
        self.path(otherwise, join, contexts, None)?;
        if !branches.is_empty() {
            self.indent -= 1;
            self.line("}");
        }
        Ok(())
    }

    fn loop_body(
        &mut self,
        header: usize,
        contexts: &[usize],
        stop: usize,
    ) -> Result<usize, Error> {
        let exits = self.graph.loops[&header].exits.clone();
        if exits.len() > 1 {
            self.line(&format!("uint metal_oxide_exit_{header} = 0u;"));
        }
        self.line("while (true) {");
        self.indent += 1;
        let mut inner = contexts.to_vec();
        inner.push(header);
        self.path(header, self.function.blocks.len(), &inner, Some(header))?;
        self.indent -= 1;
        self.line("}");
        if exits.len() == 1 {
            return Ok(exits[0]);
        }
        let join = if let Some(&parent) = contexts.last() {
            self.graph.loop_join(&exits, parent, stop)
        } else {
            self.graph.postdominators.closest_common(&exits)
        };
        let branches = exits[..exits.len() - 1]
            .iter()
            .map(|&target| (format!("metal_oxide_exit_{header} == {target}u"), target))
            .collect::<Vec<_>>();
        self.branches(&branches, *exits.last().unwrap(), join, contexts)?;
        Ok(join)
    }
}
