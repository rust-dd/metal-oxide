use metal_oxide_codegen::Codegen;
use metal_oxide_ir::*;
use std::time::{Duration, Instant};

#[test]
fn early_return_joins_do_not_enumerate_diamond_paths() {
    let source = SourceLocation {
        file: "join_cost.rs".into(),
        line: 1,
        column: 1,
    };
    let block = |terminator| Block {
        statements: vec![],
        terminator,
        source: source.clone(),
    };
    let mut blocks = vec![
        block(Terminator::Return),
        block(Terminator::Return),
        block(Terminator::Goto(1)),
    ];
    let mut arms = Vec::new();
    // Each arm has millions of paths through a small shared-tail DAG.
    for _ in 0..2 {
        let mut next = 2;
        for _ in 0..22 {
            let then_block = blocks.len();
            blocks.push(block(Terminator::Goto(next)));
            let else_block = blocks.len();
            blocks.push(block(Terminator::Goto(next)));
            next = blocks.len();
            blocks.push(block(Terminator::Branch {
                condition: Operand::local(2),
                then_block,
                else_block,
            }));
        }
        arms.push(blocks.len());
        blocks.push(block(Terminator::Branch {
            condition: Operand::local(2),
            then_block: 1,
            else_block: next,
        }));
    }
    blocks[0] = block(Terminator::Branch {
        condition: Operand::local(1),
        then_block: arms[0],
        else_block: arms[1],
    });
    let module = Module {
        types: TypeTable::default(),
        functions: vec![Function {
            name: "join_cost".into(),
            kernel: false,
            required_block: None,
            parameters: 2,
            locals: vec![
                Type::Unit,
                Type::Scalar(Scalar::Bool),
                Type::Scalar(Scalar::Bool),
            ],
            blocks,
            source,
        }],
    };
    let codegen = Codegen::new(&module).unwrap();
    let start = Instant::now();
    let output = codegen.emit().unwrap();
    assert!(output.len() < 10_000);
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "join search took {:?}",
        start.elapsed()
    );
}
