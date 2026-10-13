use crate::{
    checks::CheckTable,
    control::StructuredGraph,
    expressions::{expression, operand},
    gates::{GATE, Gates},
    unused::Used,
};
use metal_oxide_artifact::MathMode;
use metal_oxide_ir::*;
use std::{collections::HashMap, fmt::Write};

pub(crate) struct Context<'a> {
    pub(crate) module: &'a Module,
    pub(crate) function: &'a Function,
    pub(crate) id: usize,
    pub(crate) analyses: &'a [FunctionAnalysis],
    pub(crate) checks: &'a CheckTable,
    pub(crate) math_mode: MathMode,
}

pub(crate) fn body(
    context: Context<'_>,
    graph: StructuredGraph<'_>,
    used: &Used,
    output: &mut String,
) -> Result<(), Error> {
    let Context {
        module,
        function,
        id,
        analyses,
        checks,
        math_mode,
    } = context;
    Emitter {
        module,
        function,
        id,
        gates: Gates::new(function, id, analyses, &graph),
        graph,
        used,
        analyses,
        checks,
        math_mode,
        output,
        visits: HashMap::new(),
        indent: 1,
    }
    .path(0, function.blocks.len(), &[], None)
}

struct Emitter<'a> {
    module: &'a Module,
    function: &'a Function,
    id: usize,
    gates: Gates,
    graph: StructuredGraph<'a>,
    used: &'a Used,
    analyses: &'a [FunctionAnalysis],
    checks: &'a CheckTable,
    math_mode: MathMode,
    output: &'a mut String,
    visits: HashMap<usize, usize>,
    indent: usize,
}

impl Emitter<'_> {
    fn line(&mut self, value: &str) {
        writeln!(self.output, "{}{value}", "    ".repeat(self.indent)).unwrap();
    }

    /// Leaves the function after a failed check; helpers return a zero value their callers ignore.
    fn exit(&self) -> &'static str {
        if self.function.locals[0] == Type::Unit {
            "return;"
        } else {
            "return {};"
        }
    }

    fn runtime_check(&self, block: usize) -> bool {
        self.analyses[self.id].checks.contains(&block)
    }

    /// A panic the proofs show is never reached; branches into it emit nothing.
    fn dead(&self, block: usize) -> bool {
        matches!(self.function.blocks[block].terminator, Terminator::Panic(_))
            && !self.runtime_check(block)
    }

    fn fail(&mut self, block: usize) {
        let check = self.checks.id(self.id, block);
        self.line(&format!("metal_oxide_fail(metal_oxide_ctx, {check}u);"));
        let exit = self.exit();
        self.line(exit);
    }

    fn assertion(&mut self, block: usize) {
        let Terminator::Assert {
            condition,
            expected,
            ..
        } = &self.function.blocks[block].terminator
        else {
            unreachable!()
        };
        let condition = operand(self.module, self.function, condition);
        let failed = if *expected {
            format!("!({condition})")
        } else {
            condition
        };
        self.line(&format!("if ({failed}) {{"));
        self.indent += 1;
        self.fail(block);
        self.indent -= 1;
        self.line("}");
    }

    /// Emits a block's statements; a call to a fallible helper is followed by an early exit.
    fn statements(&mut self, current: usize) -> Result<(), Error> {
        let block = &self.function.blocks[current];
        let visits = self.visits.entry(current).or_default();
        *visits += 1;
        if *visits > 8
            || (*visits > 1
                && block
                    .statements
                    .iter()
                    .any(|s| s.value.is_cooperative(self.analyses)))
        {
            return Err(Error::new(
                &block.source,
                "control flow requires excessive or cooperative block duplication",
            ));
        }
        for (index, statement) in block.statements.iter().enumerate() {
            if self.gates.before_statement(current, index) {
                self.line(GATE);
            }
            if matches!(statement.value, Expression::Uninit(_))
                || !self.used.statement(current, index)
            {
                continue;
            }
            let value = expression(
                self.module,
                self.function,
                &statement.value,
                &statement.source,
                self.math_mode,
            )?;
            if self.function.locals[statement.destination.local] == Type::Unit {
                if !value.is_empty() {
                    self.line(&format!("{value};"));
                }
            } else {
                self.line(&format!(
                    "{} = {value};",
                    crate::expressions::place(self.module, self.function, &statement.destination)
                ));
            }
            if statement.value.is_fallible(self.analyses) {
                let exit = self.exit();
                self.line(&format!("if (*metal_oxide_ctx.failed) {{ {exit} }}"));
            }
        }
        if self.gates.before_statement(current, block.statements.len()) {
            self.line(GATE);
        }
        Ok(())
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
            if let Terminator::Panic(_) = self.function.blocks[current].terminator {
                // A panic leaves the function, so it needs no loop exit or join.
                self.statements(current)?;
                if self.runtime_check(current) {
                    self.fail(current);
                } else {
                    let exit = self.exit();
                    self.line(exit);
                }
                return Ok(());
            }
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
            self.statements(current)?;
            let block = &self.function.blocks[current];
            match &block.terminator {
                Terminator::Goto(target) => current = *target,
                Terminator::Assert { target, .. } => {
                    if self.runtime_check(current) {
                        self.assertion(current);
                    }
                    current = *target;
                }
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
                } if !self.dead(*then_block) && !self.dead(*else_block) => {
                    let join = self.branch_join(current, contexts, stop);
                    self.branches(
                        &[(operand(self.module, self.function, condition), *then_block)],
                        *else_block,
                        join,
                        contexts,
                    )?;
                    current = join;
                }
                Terminator::Branch {
                    then_block,
                    else_block,
                    ..
                } => {
                    current = if self.dead(*then_block) {
                        *else_block
                    } else {
                        *then_block
                    };
                }
                Terminator::Switch {
                    discriminant,
                    cases,
                    otherwise,
                } => {
                    let join = self.branch_join(current, contexts, stop);
                    let live = cases
                        .iter()
                        .filter(|(_, target)| !self.dead(*target))
                        .collect::<Vec<_>>();
                    let (live, otherwise) = if !self.dead(*otherwise) {
                        (&live[..], *otherwise)
                    } else if let Some(((_, last), rest)) = live.split_last() {
                        (rest, *last)
                    } else {
                        let exit = self.exit();
                        self.line(exit);
                        return Ok(());
                    };
                    let mut groups = Vec::<(usize, Vec<Constant>)>::new();
                    for &&(value, target) in live {
                        if target == otherwise {
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
                    self.branches(&branches, otherwise, join, contexts)?;
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
            self.graph
                .branch_join(&self.function.blocks[block].terminator.successors(), stop)
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
        if self.gates.before_loop(header) {
            self.line(GATE);
        }
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
