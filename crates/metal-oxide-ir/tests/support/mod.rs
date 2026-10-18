#![allow(dead_code)]

use metal_oxide_ir::*;

pub const U32: Type = Type::Scalar(Scalar::U32);
pub const BOOL: Type = Type::Scalar(Scalar::Bool);
pub const MATRIX: Type = Type::Matrix(Scalar::F32);

pub fn source() -> SourceLocation {
    SourceLocation {
        file: "kernel.rs".into(),
        line: 3,
        column: 5,
    }
}

pub fn assign(local: usize, value: Expression) -> Statement {
    Statement {
        destination: Place::local(local),
        value,
        source: source(),
    }
}

pub fn block(statements: Vec<Statement>, terminator: Terminator) -> Block {
    Block {
        statements,
        terminator,
        source: source(),
    }
}

pub fn uint(value: u32) -> Operand {
    Operand::Constant(Constant::U32(value))
}

pub fn output() -> Type {
    Type::Buffer {
        element: Element::Scalar(Scalar::F32),
        access: Access::Write,
        address_space: AddressSpace::Device,
    }
}

pub fn below(local: usize, limit: u32) -> Expression {
    Expression::Binary(BinaryOp::Lt, Operand::local(local), uint(limit))
}

pub fn branch(condition: usize, then_block: usize, else_block: usize) -> Terminator {
    Terminator::Branch {
        condition: Operand::local(condition),
        then_block,
        else_block,
    }
}

pub fn barrier() -> Statement {
    assign(0, Expression::ThreadgroupBarrier)
}

pub fn call(function: usize, argument: usize) -> Statement {
    assign(
        0,
        Expression::Call {
            function,
            arguments: vec![Operand::local(argument)],
        },
    )
}

/// A function of one parameter; a kernel runs in blocks of 64 threads.
pub fn function(name: &str, kernel: bool, locals: Vec<Type>, blocks: Vec<Block>) -> Function {
    Function {
        name: name.into(),
        kernel,
        required_block: kernel.then_some([64, 1, 1]),
        requires: Vec::new(),
        parameters: 1,
        locals,
        blocks,
        source: source(),
    }
}

/// A kernel over the locals `[unit, output, coordinate, condition, matrix, u32]`.
pub fn kernel(blocks: Vec<Block>) -> Function {
    let locals = vec![Type::Unit, output(), U32, BOOL, MATRIX, U32];
    function("kernel", true, locals, blocks)
}

pub fn module(functions: Vec<Function>) -> Module {
    Module {
        types: TypeTable::default(),
        functions,
    }
}

pub fn splat() -> Expression {
    Expression::Matrix {
        op: MatrixOp::Splat,
        arguments: vec![Operand::Constant(Constant::F32(0))],
    }
}

/// Stores the matrix in local `value` at `row` of the output in local 1.
pub fn store(row: Operand, value: usize) -> Statement {
    let arguments = vec![
        Operand::local(1),
        row,
        uint(0),
        uint(8),
        Operand::local(value),
    ];
    assign(
        0,
        Expression::Matrix {
            op: MatrixOp::Store { transposed: false },
            arguments,
        },
    )
}

/// Splats a matrix into local 4 and stores it to the kernel's output.
pub fn matrices() -> Vec<Statement> {
    vec![assign(4, splat()), store(uint(0), 4)]
}
