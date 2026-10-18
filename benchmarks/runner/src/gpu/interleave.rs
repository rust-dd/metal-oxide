use super::*;

#[cfg(test)]
mod tests;

/// A variant of one case, set up once: each call measures and checks one sample.
pub(super) trait Candidate {
    /// Runs one sample, kept for the report unless it belongs to the warmup.
    fn sample(&mut self, keep: bool) -> Result<()>;

    fn finish(self: Box<Self>) -> Result<Case>;
}

/// The candidates of `round` in turn: each once, starting with the one at `round` modulo `count`.
fn round_order(count: usize, round: usize) -> impl Iterator<Item = usize> {
    (0..count).map(move |offset| (round + offset) % count)
}

/// Gives every candidate one sample per warmup and timed round, so drift during the run reaches
/// them alike; returns their cases in their order.
pub(super) fn interleave(
    config: &Config,
    mut candidates: Vec<Box<dyn Candidate + '_>>,
) -> Result<Vec<Case>> {
    for round in 0..config.warmup + config.samples {
        for index in round_order(candidates.len(), round) {
            candidates[index].sample(round >= config.warmup)?;
        }
    }
    candidates
        .into_iter()
        .map(|candidate| candidate.finish())
        .collect()
}

/// Builds the rust and then the msl candidate of one case with `build` and interleaves them.
pub(super) fn rust_and_msl<'a, C: Candidate + 'a>(
    config: &Config,
    build: impl Fn(&str) -> Result<C>,
) -> Result<Vec<Case>> {
    let mut candidates = Vec::<Box<dyn Candidate + 'a>>::new();
    for variant in ["rust", "msl"] {
        candidates.push(Box::new(build(variant)?));
    }
    interleave(config, candidates)
}
