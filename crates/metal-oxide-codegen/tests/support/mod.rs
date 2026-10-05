use metal_oxide_ir::*;
use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

pub fn source() -> SourceLocation {
    SourceLocation {
        file: "kernel.rs".into(),
        line: 3,
        column: 1,
    }
}

pub fn module(parameters: usize, locals: Vec<Type>, blocks: Vec<Block>) -> Module {
    Module {
        functions: vec![Function {
            name: "helper".into(),
            kernel: false,
            parameters,
            locals,
            blocks,
            source: source(),
        }],
    }
}

pub fn block(statements: Vec<(usize, Expression)>, terminator: Terminator) -> Block {
    Block {
        statements: statements
            .into_iter()
            .map(|(destination, value)| Statement {
                destination,
                value,
                source: source(),
            })
            .collect(),
        terminator,
        source: source(),
    }
}

pub fn local(id: usize) -> Operand {
    Operand::local(id)
}
pub fn uint(value: u32) -> Operand {
    Operand::Constant(Constant::U32(value))
}

pub fn execute(module: &Module, main: &str) -> String {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../target/codegen-tests/{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("metal_stdlib"), include_str!("metal_stdlib")).unwrap();
    let msl = metal_oxide_codegen::emit(module).unwrap();
    let program = format!(
        "{msl}\n#include <iostream>\nint main() {{ metal_oxide_context ctx = {{ {{0,0,0}}, {{0,0,0}}, {{1,1,1}}, {{1,1,1}} }}; {main} }}\n"
    );
    std::fs::write(directory.join("main.cpp"), program).unwrap();
    let output = Command::new("clang++")
        .args(["-std=c++17", "-Wno-unknown-attributes"])
        .arg("-I")
        .arg(&directory)
        .arg(directory.join("main.cpp"))
        .arg("-o")
        .arg(directory.join("run"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(directory.join("run")).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}
