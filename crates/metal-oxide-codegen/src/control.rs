use metal_oxide_ir::{ControlFlowGraph, Dominators, Error, Function};
use std::collections::{BTreeSet, HashMap, VecDeque};

pub(crate) struct Loop {
    pub(crate) members: BTreeSet<usize>,
    pub(crate) exits: Vec<usize>,
}

pub(crate) struct StructuredGraph {
    pub(crate) cfg: ControlFlowGraph,
    pub(crate) loops: HashMap<usize, Loop>,
    forward: Vec<Vec<usize>>,
    pub(crate) postdominators: Dominators,
}

impl StructuredGraph {
    pub(crate) fn new(function: &Function) -> Result<Self, Error> {
        let n = function.blocks.len();
        let cfg = ControlFlowGraph::new(function)?;
        let all = cfg.reachable_blocks().collect::<BTreeSet<_>>();
        let dominators = cfg.dominators();
        let mut loop_members = HashMap::<usize, BTreeSet<usize>>::new();
        let mut forward = vec![Vec::new(); n];
        let mut incoming = vec![0; n];
        for &id in &all {
            for &target in cfg.successors(id) {
                if dominators.dominates(target, id) {
                    let members = loop_members
                        .entry(target)
                        .or_insert_with(|| BTreeSet::from([target]));
                    let mut pending = vec![id];
                    while let Some(node) = pending.pop() {
                        if members.insert(node) {
                            pending.extend(cfg.predecessors(node));
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
                .flat_map(|&id| cfg.successors(id))
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
        let postdominators = cfg.postdominators();
        Ok(Self {
            cfg,
            loops,
            forward,
            postdominators,
        })
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
            return self.cfg.exit_block();
        };
        let join = self.postdominators.closest_common(targets);
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
