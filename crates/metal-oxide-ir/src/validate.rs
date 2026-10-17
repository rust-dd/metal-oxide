use std::collections::HashSet;

use crate::{
    typing::{expression_type, place_info},
    *,
};

/// Checks types, initialization, control flow, participation, and numerical safety.
pub fn validate(module: &Module) -> Result<Vec<FunctionAnalysis>, Error> {
    let type_source = SourceLocation {
        file: "<types>".into(),
        line: 0,
        column: 0,
    };
    module.types.validate(
        module
            .functions
            .first()
            .map_or(&type_source, |function| &function.source),
    )?;
    let mut names = HashSet::new();
    for function in &module.functions {
        if let Some(shape) = function.required_block
            && (!function.kernel
                || shape
                    .into_iter()
                    .try_fold(1_u32, |n, axis| {
                        (axis != 0).then(|| n.checked_mul(axis)).flatten()
                    })
                    .is_none())
        {
            return Err(Error::new(&function.source, "invalid kernel block shape"));
        }
        if function.locals.len() <= function.parameters || function.blocks.is_empty() {
            return Err(Error::new(
                &function.source,
                "function requires return/parameter locals and an entry block",
            ));
        }
        if function.kernel {
            if !names.insert(&function.name) {
                return Err(Error::new(&function.source, "duplicate kernel name"));
            }
            if function.locals[0] != Type::Unit {
                return Err(Error::new(&function.source, "kernels must return unit"));
            }
            for ty in &function.locals[1..=function.parameters] {
                let valid = match *ty {
                    Type::Buffer {
                        element,
                        access: Access::Read | Access::Write | Access::ReadWrite | Access::Atomic,
                        address_space: AddressSpace::Device,
                    }
                    | Type::Buffer {
                        element,
                        access: Access::ReadWrite,
                        address_space: AddressSpace::Threadgroup,
                    } => module.types.is_abi_value(element.ty()),
                    _ => module.types.is_abi_value(*ty),
                };
                if !valid {
                    return Err(Error::new(
                        &function.source,
                        "unsupported kernel parameter type",
                    ));
                }
            }
        }
        for ty in &function.locals {
            if let Type::Aggregate(id)
            | Type::Buffer {
                element: Element::Aggregate(id),
                ..
            } = ty
                && module.types.get(*id).is_none()
            {
                return Err(Error::new(&function.source, "invalid aggregate type"));
            }
            if matches!(ty, Type::Pointer(_)) && module.types.pointer(*ty).is_none() {
                return Err(Error::new(&function.source, "invalid pointer type"));
            }
            if matches!(
                ty,
                Type::Buffer {
                    access: Access::Read | Access::Write,
                    address_space: AddressSpace::Threadgroup,
                    ..
                }
            ) {
                return Err(Error::new(
                    &function.source,
                    "buffer access mode does not match its address space",
                ));
            }
            if matches!(
                ty,
                Type::Buffer {
                    element: Element::Scalar(Scalar::Bool | Scalar::Usize | Scalar::Isize),
                    ..
                } | Type::Buffer {
                    element: Element::Scalar(
                        Scalar::F16
                            | Scalar::BF16
                            | Scalar::U8
                            | Scalar::U16
                            | Scalar::I8
                            | Scalar::I16
                    ) | Element::Aggregate(_),
                    access: Access::Atomic,
                    ..
                } | Type::Checked(Scalar::Bool | Scalar::F16 | Scalar::BF16)
            ) {
                return Err(Error::new(&function.source, "unsupported local type"));
            }
            if let Type::Matrix(scalar) = ty
                && !crate::is_matrix_element(*scalar)
            {
                return Err(Error::new(&function.source, "unsupported local type"));
            }
        }
    }
    let mut graphs = Vec::with_capacity(module.functions.len());
    for function in &module.functions {
        let graph = ControlFlowGraph::new(function)?;
        for block in &function.blocks {
            for statement in &block.statements {
                if !place_info(module, function, &statement.destination, &statement.source)?
                    .writable
                {
                    return Err(Error::new(
                        &statement.source,
                        "write through a shared reference",
                    ));
                }
                buffer_slice(module, function, statement)?;
                let expected =
                    place_type(module, function, &statement.destination, &statement.source)?;
                let actual =
                    expression_type(module, function, &statement.value, &statement.source)?;
                if expected != actual {
                    return Err(Error::new(
                        &statement.source,
                        format!(
                            "assignment type mismatch: expected {expected:?}, found {actual:?}"
                        ),
                    ));
                }
            }
            if let Terminator::Branch { condition, .. } | Terminator::Assert { condition, .. } =
                &block.terminator
                && operand_type(module, function, condition, &block.source)?
                    != Type::Scalar(Scalar::Bool)
            {
                return Err(Error::new(
                    &block.source,
                    "branch/assert condition must have bool type",
                ));
            }
            if let Terminator::Switch {
                discriminant,
                cases,
                ..
            } = &block.terminator
            {
                let ty = operand_type(module, function, discriminant, &block.source)?;
                if !matches!(ty, Type::Scalar(scalar) if scalar.is_integer()) {
                    return Err(Error::new(
                        &block.source,
                        "switch discriminant must be an integer",
                    ));
                }
                let mut values = HashSet::new();
                for &(value, _) in cases {
                    if value.ty() != ty {
                        return Err(Error::new(
                            &block.source,
                            "switch case type must match discriminant",
                        ));
                    }
                    if !values.insert(value) {
                        return Err(Error::new(&block.source, "duplicate switch case value"));
                    }
                }
            }
        }
        crate::dataflow::initialized(module, function, &graph)?;
        graphs.push(FunctionAnalysis {
            dominators: graph.dominators(),
            postdominators: graph.postdominators(),
            cfg: graph,
            cooperative: false,
            checks: Default::default(),
            fallible: false,
            effects: false,
            one_dimensional: false,
        });
    }
    let order = acyclic_calls(module)?;
    let checks = crate::proofs::prove_numerics(module, &graphs)?;
    for (analysis, checks) in graphs.iter_mut().zip(checks) {
        analysis.checks = checks;
    }
    for &id in &order {
        let function = &module.functions[id];
        let statements = || {
            graphs[id]
                .cfg
                .reachable_blocks()
                .flat_map(|block| &function.blocks[block].statements)
        };
        let fallible = !graphs[id].checks.is_empty()
            || statements().any(|statement| statement.value.is_fallible(&graphs));
        let effects = statements().any(|statement| statement.has_effect(module, function, &graphs));
        let one_dimensional = statements().any(|statement| match statement.value {
            Expression::ThreadIndex => true,
            Expression::Call { function, .. } => graphs[function].one_dimensional,
            _ => false,
        });
        graphs[id].fallible = fallible;
        graphs[id].effects = effects;
        graphs[id].one_dimensional = one_dimensional;
    }
    crate::uniform::validate(module, &mut graphs, &order)?;
    Ok(graphs)
}

fn acyclic_calls(module: &Module) -> Result<Vec<usize>, Error> {
    let mut incoming = vec![0_usize; module.functions.len()];
    let mut edges = vec![Vec::new(); module.functions.len()];
    for (id, function) in module.functions.iter().enumerate() {
        for statement in function.blocks.iter().flat_map(|b| &b.statements) {
            if let Expression::Call {
                function: callee, ..
            } = statement.value
            {
                incoming[callee] += 1;
                edges[id].push(callee);
            }
        }
    }
    let mut ready = incoming
        .iter()
        .enumerate()
        .filter_map(|(id, &n)| (n == 0).then_some(id))
        .collect::<Vec<_>>();
    let mut order = Vec::new();
    while let Some(id) = ready.pop() {
        order.push(id);
        for &callee in &edges[id] {
            incoming[callee] -= 1;
            if incoming[callee] == 0 {
                ready.push(callee);
            }
        }
    }
    if order.len() == module.functions.len() {
        order.reverse();
        Ok(order)
    } else {
        let id = incoming.iter().position(|&n| n > 0).unwrap();
        Err(Error::new(
            &module.functions[id].source,
            "recursion is not supported",
        ))
    }
}

/// A buffer's length is the one the runtime binds for a kernel parameter, so a buffer slice or
/// length must name that parameter itself, which the kernel never reassigns.
fn buffer_slice(module: &Module, function: &Function, statement: &Statement) -> Result<(), Error> {
    let (Expression::Slice(data) | Expression::Len(data)) = &statement.value else {
        return Ok(());
    };
    if !matches!(
        operand_type(module, function, data, &statement.source)?,
        Type::Buffer { .. }
    ) {
        return Ok(());
    }
    let parameter = match data {
        Operand::Place(place) if place.projection.is_empty() => place.local,
        _ => usize::MAX,
    };
    let assigned = function
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .any(|other| other.destination.local == parameter);
    if !function.kernel || !(1..=function.parameters).contains(&parameter) || assigned {
        return Err(Error::new(
            &statement.source,
            "buffer lengths and slices require an unmodified kernel buffer parameter",
        ));
    }
    Ok(())
}
