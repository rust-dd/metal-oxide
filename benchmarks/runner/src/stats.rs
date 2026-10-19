use serde::Serialize;

#[derive(Debug, Serialize)]
pub(crate) struct Summary {
    pub min: u64,
    pub p10: u64,
    pub median: u64,
    pub p90: u64,
    pub max: u64,
    pub mean: f64,
}

impl Summary {
    pub(crate) fn new(samples: &[u64]) -> Result<Self, &'static str> {
        if samples.is_empty() {
            return Err("at least one sample is required");
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let upper = sorted.len() / 2;
        let lower = (sorted.len() - 1) / 2;
        Ok(Self {
            min: sorted[0],
            p10: sorted[(sorted.len() - 1) / 10],
            median: sorted[lower] + (sorted[upper] - sorted[lower]) / 2,
            p90: sorted[((sorted.len() - 1) * 9).div_ceil(10)],
            max: *sorted.last().unwrap(),
            mean: sorted.iter().map(|&n| n as f64).sum::<f64>() / sorted.len() as f64,
        })
    }
}

#[cfg(test)]
mod tests;
