use metal_oxide_ir::{Error, Function};
use std::collections::{BTreeSet, HashMap};

pub(crate) struct Loop {
    pub(crate) members: BTreeSet<usize>,
    pub(crate) exit: usize,
}

pub(crate) struct Graph {
    pub(crate) reachable: Vec<bool>,
    pub(crate) loops: HashMap<usize, Loop>,
    postdominators: Vec<BTreeSet<usize>>,
}

impl Graph {
    pub(crate) fn new(function: &Function) -> Result<Self, Error> {
        let n = function.blocks.len();
        let successors = function
            .blocks
            .iter()
            .map(|b| b.terminator.successors())
            .collect::<Vec<_>>();
        let mut reachable = vec![false; n];
        let mut pending = vec![0];
        while let Some(id) = pending.pop() {
            if std::mem::replace(&mut reachable[id], true) {
                continue;
            }
            pending.extend(&successors[id]);
        }
        let all = (0..n).filter(|&i| reachable[i]).collect::<BTreeSet<_>>();
        let mut predecessors = vec![Vec::new(); n];
        for &id in &all {
            for &target in &successors[id] {
                predecessors[target].push(id);
            }
        }
        let mut dominators = vec![all.clone(); n];
        dominators[0] = BTreeSet::from([0]);
        loop {
            let mut changed = false;
            for &id in &all {
                if id == 0 {
                    continue;
                }
                let mut next = all.clone();
                for &pred in &predecessors[id] {
                    next.retain(|v| dominators[pred].contains(v));
                }
                next.insert(id);
                changed |= next != dominators[id];
                dominators[id] = next;
            }
            if !changed {
                break;
            }
        }
        let mut loop_members = HashMap::<usize, BTreeSet<usize>>::new();
        let mut forward = vec![Vec::new(); n];
        let mut incoming = vec![0; n];
        for &id in &all {
            for &target in &successors[id] {
                if dominators[id].contains(&target) {
                    let members = loop_members
                        .entry(target)
                        .or_insert_with(|| BTreeSet::from([target]));
                    let mut pending = vec![id];
                    while let Some(node) = pending.pop() {
                        if members.insert(node) {
                            pending.extend(&predecessors[node]);
                        }
                    }
                } else {
                    forward[id].push(target);
                    incoming[target] += 1;
                }
            }
        }
        let mut ready = all
            .iter()
            .copied()
            .filter(|&id| incoming[id] == 0)
            .collect::<Vec<_>>();
        let mut count = 0;
        while let Some(id) = ready.pop() {
            count += 1;
            for &target in &forward[id] {
                incoming[target] -= 1;
                if incoming[target] == 0 {
                    ready.push(target);
                }
            }
        }
        if count != all.len() {
            return Err(Error::new(
                &function.source,
                "irreducible control flow is not supported",
            ));
        }
        let mut loops = HashMap::new();
        for (header, members) in loop_members {
            let exits = members
                .iter()
                .flat_map(|&id| &successors[id])
                .copied()
                .filter(|id| !members.contains(id))
                .collect::<BTreeSet<_>>();
            if exits.len() != 1 {
                return Err(Error::new(
                    &function.blocks[header].source,
                    "loops require a single exit",
                ));
            }
            loops.insert(
                header,
                Loop {
                    members,
                    exit: *exits.first().unwrap(),
                },
            );
        }
        let mut universe = all.clone();
        universe.insert(n);
        let mut postdominators = vec![universe.clone(); n + 1];
        postdominators[n] = BTreeSet::from([n]);
        loop {
            let mut changed = false;
            for &id in all.iter().rev() {
                let mut next = universe.clone();
                let targets = if successors[id].is_empty() {
                    vec![n]
                } else {
                    successors[id].clone()
                };
                for target in targets {
                    next.retain(|v| postdominators[target].contains(v));
                }
                next.insert(id);
                changed |= next != postdominators[id];
                postdominators[id] = next;
            }
            if !changed {
                break;
            }
        }
        Ok(Self {
            reachable,
            loops,
            postdominators,
        })
    }

    pub(crate) fn join(&self, block: usize) -> usize {
        self.postdominators[block]
            .iter()
            .copied()
            .filter(|&id| id != block)
            .max_by_key(|&id| self.postdominators[id].len())
            .unwrap_or(self.reachable.len())
    }
}
