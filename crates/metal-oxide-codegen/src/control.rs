use metal_oxide_ir::{ControlFlowGraph, Dominators, Error, Function, FunctionAnalysis};
use std::collections::{BTreeSet, HashMap, VecDeque};

pub(crate) struct Loop {
    pub(crate) members: BTreeSet<usize>,
    pub(crate) exits: Vec<usize>,
}

pub(crate) struct StructuredGraph<'a> {
    pub(crate) cfg: &'a ControlFlowGraph,
    pub(crate) loops: HashMap<usize, Loop>,
    forward: Vec<Vec<usize>>,
    pub(crate) postdominators: &'a Dominators,
}

#[derive(Clone, Copy)]
enum Reachability {
    Visiting,
    Resolved(bool),
}

impl<'a> StructuredGraph<'a> {
    pub(crate) fn branch_join(&self, targets: &[usize], stop: usize) -> usize {
        let join = self.postdominators.closest_common(targets);
        if join != self.cfg.exit_block() && !self.cfg.successors(join).is_empty() {
            return join;
        }
        let mut distance = HashMap::<usize, (usize, usize)>::new();
        for &target in targets {
            let mut pending = VecDeque::from([(target, 0)]);
            let mut visited = BTreeSet::new();
            while let Some((id, depth)) = pending.pop_front() {
                if !visited.insert(id) {
                    continue;
                }
                distance
                    .entry(id)
                    .and_modify(|(count, d)| {
                        *count += 1;
                        *d = (*d).max(depth);
                    })
                    .or_insert((1, depth));
                if id != stop {
                    pending.extend(self.cfg.successors(id).iter().map(|&id| (id, depth + 1)));
                }
            }
        }
        let mut candidates = distance
            .into_iter()
            .filter(|(id, _)| !self.cfg.successors(*id).is_empty())
            .collect::<Vec<_>>();
        candidates.sort_by_key(|&(id, (count, distance))| (std::cmp::Reverse(count), distance, id));
        for (candidate, _) in candidates {
            let mut results = vec![None; self.cfg.exit_block()];
            if targets
                .iter()
                .all(|&target| self.returns_or_reaches(target, candidate, stop, &mut results))
            {
                return candidate;
            }
        }
        join
    }

    fn returns_or_reaches(
        &self,
        id: usize,
        candidate: usize,
        stop: usize,
        results: &mut [Option<Reachability>],
    ) -> bool {
        if id == candidate || self.cfg.successors(id).is_empty() {
            return true;
        }
        if id == stop {
            return false;
        }
        match results[id] {
            Some(Reachability::Resolved(valid)) => return valid,
            Some(Reachability::Visiting) => return false,
            None => {}
        }
        results[id] = Some(Reachability::Visiting);
        let valid = self
            .cfg
            .successors(id)
            .iter()
            .all(|&next| self.returns_or_reaches(next, candidate, stop, results));
        results[id] = Some(Reachability::Resolved(valid));
        valid
    }

    pub(crate) fn new(function: &Function, analysis: &'a FunctionAnalysis) -> Result<Self, Error> {
        let n = function.blocks.len();
        let cfg = &analysis.cfg;
        let all = cfg.reachable_blocks().collect::<BTreeSet<_>>();
        let dominators = &analysis.dominators;
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
        let postdominators = &analysis.postdominators;
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
