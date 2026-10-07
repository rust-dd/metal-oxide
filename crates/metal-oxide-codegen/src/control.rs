use metal_oxide_ir::{Error, Function};
use std::collections::{BTreeSet, HashMap, VecDeque};

pub(crate) struct Loop {
    pub(crate) members: BTreeSet<usize>,
    pub(crate) exits: Vec<usize>,
}

pub(crate) struct Graph {
    pub(crate) reachable: Vec<bool>,
    pub(crate) loops: HashMap<usize, Loop>,
    forward: Vec<Vec<usize>>,
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
            if exits.is_empty() {
                return Err(Error::new(
                    &function.blocks[header].source,
                    "loops require an exit",
                ));
            }
            loops.insert(
                header,
                Loop {
                    members,
                    exits: exits.into_iter().collect(),
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
            forward,
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

    pub(crate) fn common_join(&self, blocks: &[usize]) -> usize {
        let mut common = self.postdominators[blocks[0]].clone();
        for &block in &blocks[1..] {
            common.retain(|id| self.postdominators[block].contains(id));
        }
        common
            .into_iter()
            .max_by_key(|&id| self.postdominators[id].len())
            .unwrap_or(self.reachable.len())
    }

    pub(crate) fn loop_join(&self, targets: &[usize], header: usize, stop: usize) -> usize {
        let members = &self.loops[&header].members;
        let mut paths = Vec::new();
        for &target in targets {
            let mut distances = HashMap::new();
            let mut pending = VecDeque::from([(target, 0)]);
            while let Some((block, distance)) = pending.pop_front() {
                if block == header || !members.contains(&block) || distances.contains_key(&block) {
                    continue;
                }
                distances.insert(block, distance);
                // Crossing the caller's join would duplicate its continuation in a nested branch.
                if block != stop {
                    pending.extend(self.forward[block].iter().map(|&next| (next, distance + 1)));
                }
            }
            if !distances.is_empty() {
                paths.push(distances);
            }
        }
        let Some(first) = paths.first() else {
            return self.reachable.len();
        };
        let join = self.common_join(targets);
        if paths.iter().all(|path| path.contains_key(&join)) {
            return join;
        }
        let mut common = first.keys().copied().collect::<BTreeSet<_>>();
        for path in &paths[1..] {
            common.retain(|id| path.contains_key(id));
        }
        if common.is_empty() {
            if paths.iter().any(|path| path.contains_key(&stop)) {
                return stop;
            }
            return *first
                .iter()
                .min_by_key(|&(id, distance)| (*distance, *id))
                .unwrap()
                .0;
        }
        common
            .into_iter()
            .min_by_key(|id| (paths.iter().map(|path| path[id]).max().unwrap(), *id))
            .unwrap()
    }
}
