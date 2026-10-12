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
        let all = self.reachable_set(self.successors.len());
        let mut sets = vec![all.clone(); self.successors.len()];
        sets[0] = Blocks::single(self.successors.len(), 0);
        loop {
            let mut changed = false;
            for id in self.reachable_blocks().filter(|&id| id != 0) {
                let mut next = all.clone();
                for &pred in &self.predecessors[id] {
                    next.intersect(&sets[pred]);
                }
                next.insert(id);
                changed |= next != sets[id];
                sets[id] = next;
            }
            if !changed {
                break;
            }
        }
        Dominators::new(sets, 0)
    }

    /// Return blocks lead to a synthetic exit numbered after the last real block.
    pub fn postdominators(&self) -> Dominators {
        let exit = self.exit_block();
        let mut all = self.reachable_set(exit + 1);
        all.insert(exit);
        let mut sets = vec![all.clone(); exit + 1];
        sets[exit] = Blocks::single(exit + 1, exit);
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
                    next.intersect(&sets[target]);
                }
                next.insert(id);
                changed |= next != sets[id];
                sets[id] = next;
            }
            if !changed {
                break;
            }
        }
        Dominators::new(sets, exit)
    }

    fn reachable_set(&self, capacity: usize) -> Blocks {
        let mut set = Blocks::empty(capacity);
        self.reachable_blocks().for_each(|id| set.insert(id));
        set
    }
}

/// A fixed-capacity block set.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Blocks(Vec<u64>);

impl Blocks {
    fn empty(capacity: usize) -> Self {
        Self(vec![0; capacity.div_ceil(64)])
    }

    fn single(capacity: usize, id: usize) -> Self {
        let mut set = Self::empty(capacity);
        set.insert(id);
        set
    }

    fn insert(&mut self, id: usize) {
        self.0[id / 64] |= 1 << (id % 64);
    }

    fn contains(&self, id: usize) -> bool {
        self.0
            .get(id / 64)
            .is_some_and(|word| word >> (id % 64) & 1 == 1)
    }

    fn intersect(&mut self, other: &Self) {
        self.0.iter_mut().zip(&other.0).for_each(|(a, b)| *a &= b);
    }

    fn len(&self) -> usize {
        self.0.iter().map(|word| word.count_ones() as usize).sum()
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(index, &word)| {
            let mut rest = word;
            std::iter::from_fn(move || {
                (rest != 0).then(|| {
                    let bit = rest.trailing_zeros() as usize;
                    rest &= rest - 1;
                    index * 64 + bit
                })
            })
        })
    }
}

/// Dominance sets rooted at the entry or synthetic exit block.
#[derive(Debug)]
pub struct Dominators {
    sets: Vec<Blocks>,
    sizes: Vec<usize>,
    root: usize,
}

impl Dominators {
    fn new(sets: Vec<Blocks>, root: usize) -> Self {
        Self {
            sizes: sets.iter().map(Blocks::len).collect(),
            sets,
            root,
        }
    }

    pub fn dominates(&self, dominator: usize, block: usize) -> bool {
        self.sets[block].contains(dominator)
    }

    pub fn immediate(&self, block: usize) -> usize {
        self.sets[block]
            .iter()
            .filter(|&id| id != block)
            .max_by_key(|&id| self.sizes[id])
            .unwrap_or(self.root)
    }

    pub fn closest_common(&self, blocks: &[usize]) -> usize {
        let Some((&first, rest)) = blocks.split_first() else {
            return self.root;
        };
        let mut common = self.sets[first].clone();
        for &block in rest {
            common.intersect(&self.sets[block]);
        }
        common
            .iter()
            .max_by_key(|&id| self.sizes[id])
            .unwrap_or(self.root)
    }
}
