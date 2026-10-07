use std::collections::BTreeSet;

use crate::{Error, Function};

/// Block connectivity, independent of codegen's supported control-flow subset.
#[derive(Debug)]
pub struct ControlFlowGraph {
    successors: Vec<Vec<usize>>,
    predecessors: Vec<Vec<usize>>,
    reachable: Vec<bool>,
}

impl ControlFlowGraph {
    pub fn new(function: &Function) -> Result<Self, Error> {
        let n = function.blocks.len();
        if n == 0 {
            return Err(Error::new(
                &function.source,
                "function requires an entry block",
            ));
        }
        let successors = function
            .blocks
            .iter()
            .map(|b| b.terminator.successors())
            .collect::<Vec<_>>();
        for (id, targets) in successors.iter().enumerate() {
            if targets.iter().any(|&target| target >= n) {
                return Err(Error::new(
                    &function.blocks[id].source,
                    "invalid block target",
                ));
            }
        }
        let mut reachable = vec![false; n];
        let mut pending = vec![0];
        while let Some(id) = pending.pop() {
            if !std::mem::replace(&mut reachable[id], true) {
                pending.extend(&successors[id]);
            }
        }
        let mut predecessors = vec![Vec::new(); n];
        for (id, targets) in successors
            .iter()
            .enumerate()
            .filter(|(id, _)| reachable[*id])
        {
            for &target in targets {
                predecessors[target].push(id);
            }
        }
        Ok(Self {
            successors,
            predecessors,
            reachable,
        })
    }

    pub fn successors(&self, block: usize) -> &[usize] {
        &self.successors[block]
    }

    pub fn predecessors(&self, block: usize) -> &[usize] {
        &self.predecessors[block]
    }

    pub fn is_reachable(&self, block: usize) -> bool {
        self.reachable[block]
    }

    pub fn reachable_blocks(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.reachable.len()).filter(|&id| self.reachable[id])
    }

    pub fn exit_block(&self) -> usize {
        self.successors.len()
    }

    /// Requires at least one edge; `reaches(block, block)` detects a cycle.
    pub fn reaches(&self, from: usize, target: usize) -> bool {
        let mut visited = BTreeSet::new();
        let mut pending = self.successors[from].clone();
        while let Some(id) = pending.pop() {
            if id == target {
                return true;
            }
            if visited.insert(id) {
                pending.extend(&self.successors[id]);
            }
        }
        false
    }

    pub fn dominators(&self) -> Dominators {
        let all = self.reachable_blocks().collect::<BTreeSet<_>>();
        let mut sets = vec![all.clone(); self.successors.len()];
        sets[0] = BTreeSet::from([0]);
        loop {
            let mut changed = false;
            for id in self.reachable_blocks().filter(|&id| id != 0) {
                let mut next = all.clone();
                for &pred in &self.predecessors[id] {
                    next.retain(|v| sets[pred].contains(v));
                }
                next.insert(id);
                changed |= next != sets[id];
                sets[id] = next;
            }
            if !changed {
                break;
            }
        }
        Dominators { sets, root: 0 }
    }

    /// Return blocks lead to a synthetic exit numbered after the last real block.
    pub fn postdominators(&self) -> Dominators {
        let exit = self.exit_block();
        let mut all = self.reachable_blocks().collect::<BTreeSet<_>>();
        all.insert(exit);
        let mut sets = vec![all.clone(); exit + 1];
        sets[exit] = BTreeSet::from([exit]);
        loop {
            let mut changed = false;
            for id in (0..exit).rev().filter(|&id| self.reachable[id]) {
                let mut next = all.clone();
                let targets = if self.successors[id].is_empty() {
                    &[exit][..]
                } else {
                    &self.successors[id]
                };
                for &target in targets {
                    next.retain(|v| sets[target].contains(v));
                }
                next.insert(id);
                changed |= next != sets[id];
                sets[id] = next;
            }
            if !changed {
                break;
            }
        }
        Dominators { sets, root: exit }
    }
}

/// Dominance sets rooted at the entry or synthetic exit block.
#[derive(Debug)]
pub struct Dominators {
    sets: Vec<BTreeSet<usize>>,
    root: usize,
}

impl Dominators {
    pub fn dominates(&self, dominator: usize, block: usize) -> bool {
        self.sets[block].contains(&dominator)
    }

    pub fn immediate(&self, block: usize) -> usize {
        self.sets[block]
            .iter()
            .copied()
            .filter(|&id| id != block)
            .max_by_key(|&id| self.sets[id].len())
            .unwrap_or(self.root)
    }

    pub fn closest_common(&self, blocks: &[usize]) -> usize {
        let Some((&first, rest)) = blocks.split_first() else {
            return self.root;
        };
        let mut common = self.sets[first].clone();
        for &block in rest {
            common.retain(|id| self.sets[block].contains(id));
        }
        common
            .into_iter()
            .max_by_key(|&id| self.sets[id].len())
            .unwrap_or(self.root)
    }
}
