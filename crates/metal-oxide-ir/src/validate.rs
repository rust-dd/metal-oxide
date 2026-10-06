use std::collections::HashSet;

use crate::{typing::expression_type, *};

pub fn validate(module: &Module) -> Result<(), Error> {
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
                if !matches!(
                    ty,
                    Type::Scalar(
                        Scalar::F32 | Scalar::U32 | Scalar::I32 | Scalar::U8 | Scalar::U16
                    ) | Type::Buffer {
                        element: Scalar::F32 | Scalar::U32 | Scalar::I32 | Scalar::U8 | Scalar::U16,
                        access: Access::Read | Access::Write | Access::Atomic,
                        address_space: AddressSpace::Device,
                    }
                ) {
                    return Err(Error::new(
                        &function.source,
                        "unsupported kernel parameter type",
                    ));
                }
            }
        }
        for ty in &function.locals {
            if matches!(
                ty,
                Type::Buffer {
                    access: Access::ReadWrite,
                    address_space: AddressSpace::Device,
                    ..
                } | Type::Buffer {
                    access: Access::Read | Access::Write | Access::Atomic,
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
                    element: Scalar::Bool,
                    ..
                } | Type::Buffer {
                    element: Scalar::F32 | Scalar::U8 | Scalar::U16,
                    access: Access::Atomic,
                    ..
                } | Type::Checked(Scalar::Bool | Scalar::F32)
            ) {
                return Err(Error::new(&function.source, "unsupported local type"));
            }
        }
    }
    for function in &module.functions {
        for block in &function.blocks {
            for statement in &block.statements {
                let expected = function
                    .locals
                    .get(statement.destination)
                    .ok_or_else(|| Error::new(&statement.source, "invalid destination local"))?;
                let actual =
                    expression_type(module, function, &statement.value, &statement.source)?;
                if *expected != actual {
                    return Err(Error::new(
                        &statement.source,
                        format!(
                            "assignment type mismatch: expected {expected:?}, found {actual:?}"
                        ),
                    ));
                }
            }
            for target in block.terminator.successors() {
                if target >= function.blocks.len() {
                    return Err(Error::new(&block.source, "invalid block target"));
                }
            }
            if let Terminator::Branch { condition, .. } | Terminator::Assert { condition, .. } =
                &block.terminator
                && operand_type(function, condition, &block.source)? != Type::Scalar(Scalar::Bool)
            {
                return Err(Error::new(
                    &block.source,
                    "branch/assert condition must have bool type",
                ));
            }
        }
        crate::dataflow::initialized(function)?;
    }
    acyclic_calls(module)?;
    crate::uniform::validate(module)
}

fn acyclic_calls(module: &Module) -> Result<(), Error> {
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
    let mut count = 0;
    while let Some(id) = ready.pop() {
        count += 1;
        for &callee in &edges[id] {
            incoming[callee] -= 1;
            if incoming[callee] == 0 {
                ready.push(callee);
            }
        }
    }
    if count == module.functions.len() {
        Ok(())
    } else {
        let id = incoming.iter().position(|&n| n > 0).unwrap();
        Err(Error::new(
            &module.functions[id].source,
            "recursion is not supported",
        ))
    }
}
