use std::collections::BTreeSet;

use crate::*;

/// How much a value may differ between the threads of a block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    #[default]
    Block,
    /// The same within each SIMD group.
    Group,
    Lane,
}

/// The threads that reach a cooperative operation together.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Scope {
    Group,
    Block,
}

impl Scope {
    /// The widest variation of control and operands its participants tolerate.
    const fn tolerates(self) -> Level {
        match self {
            Self::Group => Level::Group,
            Self::Block => Level::Block,
        }
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
struct Dependencies {
    level: Level,
    parameters: BTreeSet<usize>,
}

impl Dependencies {
    fn of(level: Level) -> Self {
        Self {
            level,
            parameters: BTreeSet::new(),
        }
    }

    fn merge(&mut self, other: &Self) -> bool {
        let before = self.clone();
        self.level = self.level.max(other.level);
        self.parameters.extend(&other.parameters);
        *self != before
    }

    fn substitute(&self, arguments: &[Dependencies]) -> Self {
        let mut result = Self::of(self.level);
        for &parameter in &self.parameters {
            result.merge(&arguments[parameter]);
        }
        result
    }

    /// Whether `scope`'s participants agree on these values; without kernels, parameters are
    /// unresolved.
    fn admitted(&self, scope: Scope, kernels: bool) -> bool {
        self.level <= scope.tolerates() && (kernels || self.parameters.is_empty())
    }
}

/// What must be uniform for each scope at a cooperative operation or call site.
#[derive(Clone, Default)]
struct Requirements {
    block: Dependencies,
    group: Dependencies,
}

impl Requirements {
    fn require(&mut self, scope: Scope, dependencies: &Dependencies) {
        match scope {
            Scope::Block => self.block.merge(dependencies),
            Scope::Group => self.group.merge(dependencies),
        };
    }

    fn merge(&mut self, other: &Self) {
        self.block.merge(&other.block);
        self.group.merge(&other.group);
    }

    fn substitute(&self, arguments: &[Dependencies]) -> Self {
        Self {
            block: self.block.substitute(arguments),
            group: self.group.substitute(arguments),
        }
    }

    fn admitted(&self, kernels: bool) -> bool {
        self.block.admitted(Scope::Block, kernels) && self.group.admitted(Scope::Group, kernels)
    }
}

#[derive(Clone, Default)]
struct Summary {
    returns: Dependencies,
    participation: Requirements,
}

/// The scope a statement's cooperative operations need, given each function's scope.
fn scope(value: &Expression, scopes: &[Option<Scope>]) -> Option<Scope> {
    match value {
        Expression::ThreadgroupAlloc { .. }
        | Expression::ThreadgroupBarrier
        | Expression::Simd { .. } => Some(Scope::Block),
        Expression::Matrix { .. } => Some(Scope::Group),
        Expression::Call { function, .. } => scopes[*function],
        _ => None,
    }
}

/// Validates uniform participation. A failed check returns from the kernel, so a check that
/// precedes a cooperative operation must be uniform over its scope, failing on all participants.
pub(crate) fn validate(
    module: &Module,
    analyses: &mut [FunctionAnalysis],
    order: &[usize],
) -> Result<(), Error> {
    let mut scopes = vec![None; module.functions.len()];
    let mut matrices = vec![false; module.functions.len()];
    for &id in order {
        let function = &module.functions[id];
        allocations(function, &analyses[id].cfg)?;
        for block in analyses[id].cfg.reachable_blocks() {
            for statement in &function.blocks[block].statements {
                scopes[id] = scopes[id].max(scope(&statement.value, &scopes));
                matrices[id] |= match statement.value {
                    Expression::Matrix { .. } => true,
                    Expression::Call { function, .. } => matrices[function],
                    _ => false,
                };
            }
        }
        analyses[id].cooperative = scopes[id].is_some();
    }
    let continuations = continuations(module, analyses, order, &scopes);
    let mut summaries = vec![Summary::default(); module.functions.len()];
    let kernels = module.functions.iter().any(|function| function.kernel);
    for &id in order {
        let function = &module.functions[id];
        let analysis = &analyses[id];
        let (values, control) = flow(function, analysis, &summaries);
        let mut summary = Summary {
            returns: values[0].clone(),
            ..Summary::default()
        };
        let mut cooperation = scopes[id];
        for &block in &analysis.checks {
            let starts = match &function.blocks[block].terminator {
                Terminator::Assert { target, .. } => vec![*target],
                _ => analysis
                    .cfg
                    .predecessors(block)
                    .iter()
                    .flat_map(|&from| analysis.cfg.successors(from))
                    .copied()
                    .collect(),
            };
            let Some(reached) =
                reaches_cooperation(function, analysis, &scopes, &starts, continuations[id])
            else {
                continue;
            };
            let mut required = control[block].clone();
            if let Terminator::Assert { condition, .. } = &function.blocks[block].terminator {
                required.merge(&dependencies(condition, &values));
            }
            if !required.admitted(reached, kernels) {
                return Err(Error::new(
                    &function.blocks[block].source,
                    "a check before a barrier or SIMD operation must be proved or uniform; \
                     a thread that fails it would leave the others waiting",
                ));
            }
            summary.participation.require(reached, &required);
            cooperation = cooperation.max(Some(reached));
        }
        for (id, block) in function
            .blocks
            .iter()
            .enumerate()
            .filter(|(id, _)| analysis.cfg.is_reachable(*id))
        {
            for statement in &block.statements {
                let Some(needed) = scope(&statement.value, &scopes) else {
                    continue;
                };
                let mut required = Requirements::default();
                required.require(needed, &control[id]);
                match &statement.value {
                    Expression::Call {
                        function,
                        arguments,
                    } => {
                        let arguments = arguments
                            .iter()
                            .map(|v| dependencies(v, &values))
                            .collect::<Vec<_>>();
                        required.merge(&summaries[*function].participation.substitute(&arguments));
                    }
                    Expression::Simd { op, arguments } if op.uniform_control() => {
                        required.block.merge(&dependencies(&arguments[1], &values));
                    }
                    Expression::Matrix { arguments, .. } => {
                        let operands =
                            matrix_operands(arguments, &values, kernels, &statement.source)?;
                        required.group.merge(&operands);
                    }
                    _ => {}
                }
                if !required.admitted(kernels) {
                    return Err(Error::new(
                        &statement.source,
                        "cooperative operation requires uniform participation; divergent control flow is unsupported",
                    ));
                }
                summary.participation.merge(&required);
                cooperation = cooperation.max(Some(needed));
            }
        }
        summaries[id] = summary;
        scopes[id] = cooperation;
        analyses[id].cooperative = cooperation.is_some();
    }
    blocks(module, &matrices)
}

/// What a matrix operation's operands depend on; all lanes of a SIMD group must agree on them.
fn matrix_operands(
    arguments: &[Operand],
    values: &[Dependencies],
    kernels: bool,
    source: &SourceLocation,
) -> Result<Dependencies, Error> {
    let mut operands = Dependencies::default();
    for argument in arguments {
        operands.merge(&dependencies(argument, values));
    }
    if !operands.admitted(Scope::Group, kernels) {
        return Err(Error::new(
            source,
            "SIMD-group matrix operands must be uniform within the SIMD group",
        ));
    }
    Ok(operands)
}

/// Matrix operations need full SIMD groups, so their kernels fix a block of whole 32-lane groups.
fn blocks(module: &Module, matrices: &[bool]) -> Result<(), Error> {
    for (function, &uses) in module.functions.iter().zip(matrices) {
        if function.kernel
            && uses
            && !function
                .required_block
                .is_some_and(|shape| shape.iter().product::<u32>() % 32 == 0)
        {
            return Err(Error::new(
                &function.source,
                "SIMD-group matrices require a literal kernel block whose thread count is a multiple of 32",
            ));
        }
    }
    Ok(())
}

/// The widest scope a caller may still need after each function returns.
fn continuations(
    module: &Module,
    analyses: &[FunctionAnalysis],
    order: &[usize],
    scopes: &[Option<Scope>],
) -> Vec<Option<Scope>> {
    let mut result = vec![None; module.functions.len()];
    for &caller in order.iter().rev() {
        let function = &module.functions[caller];
        let analysis = &analyses[caller];
        for block in analysis.cfg.reachable_blocks() {
            let statements = &function.blocks[block].statements;
            for (index, statement) in statements.iter().enumerate() {
                let Expression::Call {
                    function: callee, ..
                } = statement.value
                else {
                    continue;
                };
                let later = statements[index + 1..]
                    .iter()
                    .filter_map(|later| scope(&later.value, scopes))
                    .max();
                let after = reaches_cooperation(
                    function,
                    analysis,
                    scopes,
                    analysis.cfg.successors(block),
                    result[caller],
                );
                result[callee] = result[callee].max(later).max(after);
            }
        }
    }
    result
}

/// The widest scope a path from `starts` needs, counting a caller's after a return.
fn reaches_cooperation(
    function: &Function,
    analysis: &FunctionAnalysis,
    scopes: &[Option<Scope>],
    starts: &[usize],
    continuation: Option<Scope>,
) -> Option<Scope> {
    let mut pending = starts.to_vec();
    let mut visited = BTreeSet::new();
    let mut reached = None;
    while let Some(id) = pending.pop() {
        if reached == Some(Scope::Block) {
            break;
        }
        if !visited.insert(id) {
            continue;
        }
        let block = &function.blocks[id];
        if block.terminator == Terminator::Return {
            reached = reached.max(continuation);
        }
        for statement in &block.statements {
            reached = reached.max(scope(&statement.value, scopes));
        }
        pending.extend(analysis.cfg.successors(id));
    }
    reached
}

fn allocations(function: &Function, graph: &ControlFlowGraph) -> Result<(), Error> {
    let mut ids = BTreeSet::new();
    for (block_id, block) in function
        .blocks
        .iter()
        .enumerate()
        .filter(|(id, _)| graph.is_reachable(*id))
    {
        for statement in &block.statements {
            if let Expression::ThreadgroupAlloc { id, .. } = statement.value {
                if !function.kernel {
                    return Err(Error::new(
                        &statement.source,
                        "threadgroup allocations must be in kernel entrypoints",
                    ));
                }
                if !ids.insert(id) {
                    return Err(Error::new(
                        &statement.source,
                        "duplicate threadgroup allocation ID",
                    ));
                }
                if graph.reaches(block_id, block_id) {
                    return Err(Error::new(
                        &statement.source,
                        "threadgroup allocations inside loops are unsupported",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn flow(
    function: &Function,
    analysis: &FunctionAnalysis,
    summaries: &[Summary],
) -> (Vec<Dependencies>, Vec<Dependencies>) {
    let graph = &analysis.cfg;
    let mut values = vec![Dependencies::default(); function.locals.len()];
    if !function.kernel {
        for (index, value) in values[1..=function.parameters].iter_mut().enumerate() {
            value.parameters.insert(index);
        }
    }
    // Writes through pointers are not tracked, so mutably borrowed locals vary by lane.
    for statement in function.blocks.iter().flat_map(|block| &block.statements) {
        if let Expression::Ref {
            place: Operand::Place(place),
            mutable: true,
        } = &statement.value
            && !place.is_indirect()
        {
            values[place.local].level = Level::Lane;
        }
    }
    let mut control = vec![Dependencies::default(); function.blocks.len()];
    loop {
        let mut changed = false;
        for (id, block) in function
            .blocks
            .iter()
            .enumerate()
            .filter(|(id, _)| graph.is_reachable(*id))
        {
            for statement in &block.statements {
                let mut value = control[id].clone();
                for projection in &statement.destination.projection {
                    if let Projection::Index(index) = projection {
                        value.merge(&values[*index]);
                    }
                }
                match &statement.value {
                    Expression::Call {
                        function,
                        arguments,
                    } => {
                        let arguments = arguments
                            .iter()
                            .map(|v| dependencies(v, &values))
                            .collect::<Vec<_>>();
                        value.merge(&summaries[*function].returns.substitute(&arguments));
                    }
                    expression => {
                        value.merge(&Dependencies::of(introduced(expression)));
                        for operand in expression.operands() {
                            value.merge(&dependencies(operand, &values));
                        }
                    }
                }
                if !statement.destination.is_indirect() {
                    changed |= values[statement.destination.local].merge(&value);
                }
            }
            if let Terminator::Branch { condition, .. }
            | Terminator::Switch {
                discriminant: condition,
                ..
            } = &block.terminator
            {
                let dependency = dependencies(condition, &values);
                if dependency == Dependencies::default() {
                    continue;
                }
                let join = analysis.postdominators.immediate(id);
                let mut pending = graph.successors(id).to_vec();
                let mut visited = BTreeSet::new();
                while let Some(node) = pending.pop() {
                    if node == join || !visited.insert(node) {
                        continue;
                    }
                    changed |= control[node].merge(&dependency);
                    pending.extend(graph.successors(node));
                }
            }
        }
        if !changed {
            return (values, control);
        }
    }
}

/// The variation an expression adds to its operands'.
fn introduced(expression: &Expression) -> Level {
    match expression {
        Expression::SimdCoordinate(SimdBuiltin::Group) => Level::Group,
        Expression::Coordinates(Builtin::ThreadIdx)
        | Expression::ThreadIndex
        | Expression::SimdCoordinate(SimdBuiltin::Lane)
        | Expression::BufferLoad { .. }
        | Expression::Atomic { .. }
        | Expression::Simd { .. }
        | Expression::Uninit(_) => Level::Lane,
        _ => Level::Block,
    }
}

fn dependencies(value: &Operand, values: &[Dependencies]) -> Dependencies {
    match value {
        Operand::Constant(_) => Dependencies::default(),
        Operand::AggregateConstant { fields, .. } => {
            let mut result = Dependencies::default();
            for field in fields {
                result.merge(&dependencies(field, values));
            }
            result
        }
        Operand::Place(place) => {
            let mut result = values[place.local].clone();
            if place.is_indirect() {
                result.merge(&Dependencies::of(Level::Lane));
            }
            for projection in &place.projection {
                if let Projection::Index(index) = projection {
                    result.merge(&values[*index]);
                }
            }
            result
        }
    }
}
