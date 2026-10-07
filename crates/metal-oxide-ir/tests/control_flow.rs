use metal_oxide_ir::{
    Block, Constant, ControlFlowGraph, Function, Operand, SourceLocation, Terminator, Type,
};

fn function(terminators: Vec<Terminator>) -> Function {
    let source = SourceLocation {
        file: "kernel.rs".into(),
        line: 1,
        column: 1,
    };
    Function {
        name: "helper".into(),
        kernel: false,
        required_block: None,
        parameters: 0,
        locals: vec![Type::Unit],
        blocks: terminators
            .into_iter()
            .map(|terminator| Block {
                statements: vec![],
                terminator,
                source: source.clone(),
            })
            .collect(),
        source,
    }
}

fn branch(then_block: usize, else_block: usize) -> Terminator {
    Terminator::Branch {
        condition: Operand::Constant(Constant::Bool(true)),
        then_block,
        else_block,
    }
}

#[test]
fn diamond_rejoins_after_both_arms_and_ignores_dead_predecessors() {
    let graph = ControlFlowGraph::new(&function(vec![
        branch(1, 2),
        Terminator::Goto(3),
        Terminator::Goto(3),
        Terminator::Return,
        Terminator::Goto(3),
    ]))
    .unwrap();
    assert!(!graph.is_reachable(4));
    assert_eq!(graph.predecessors(3), [1, 2]);
    let dominators = graph.dominators();
    assert!(dominators.dominates(0, 3));
    assert!(!dominators.dominates(1, 3));
    assert_eq!(dominators.immediate(3), 0);
    let post = graph.postdominators();
    assert_eq!(post.immediate(0), 3);
    assert_eq!(post.closest_common(&[1, 2]), 3);
}

#[test]
fn separate_returns_join_only_at_the_synthetic_exit() {
    let graph = ControlFlowGraph::new(&function(vec![
        branch(1, 2),
        Terminator::Return,
        Terminator::Goto(3),
        Terminator::Return,
    ]))
    .unwrap();
    let post = graph.postdominators();
    assert_eq!(post.immediate(0), 4);
    assert_eq!(post.closest_common(&[1, 3]), 4);
    assert!(!post.dominates(1, 0));
}

#[test]
fn backedges_are_distinct_from_zero_length_paths() {
    let graph = ControlFlowGraph::new(&function(vec![
        Terminator::Goto(1),
        branch(2, 3),
        Terminator::Goto(1),
        Terminator::Return,
    ]))
    .unwrap();
    assert!(graph.reaches(1, 1));
    assert!(!graph.reaches(0, 0));
    assert!(graph.dominators().dominates(1, 2));
    assert_eq!(graph.postdominators().immediate(1), 3);
}

#[test]
fn graph_analysis_does_not_impose_msl_structuring_rules() {
    let graph = ControlFlowGraph::new(&function(vec![
        branch(1, 2),
        branch(2, 3),
        branch(1, 3),
        Terminator::Return,
    ]))
    .unwrap();
    assert!(!graph.dominators().dominates(1, 2));
    assert!(!graph.dominators().dominates(2, 1));
    assert_eq!(graph.postdominators().immediate(0), 3);
    assert!(ControlFlowGraph::new(&function(vec![Terminator::Goto(0)])).is_ok());
}

#[test]
fn malformed_edges_and_missing_entrypoints_are_diagnostics() {
    let empty = ControlFlowGraph::new(&function(vec![])).unwrap_err();
    assert!(empty.message.contains("entry block"));
    let invalid = ControlFlowGraph::new(&function(vec![Terminator::Return, Terminator::Goto(2)]))
        .unwrap_err();
    assert_eq!(invalid.message, "invalid block target");
}
