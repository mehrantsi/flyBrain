use anyhow::{Result, bail};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventSchedule {
    targets: Vec<u32>,
    counts: Vec<u8>,
    steps: usize,
}

impl EventSchedule {
    pub fn new(
        targets: Vec<u32>,
        counts: Vec<u8>,
        steps: usize,
        neuron_count: usize,
    ) -> Result<Self> {
        if targets
            .iter()
            .any(|&target| target as usize >= neuron_count)
        {
            bail!("stimulus target is outside the connectome");
        }
        let expected = steps
            .checked_mul(targets.len())
            .ok_or_else(|| anyhow::anyhow!("stimulus dimensions overflow"))?;
        if counts.len() != expected {
            bail!("stimulus counts length does not match steps × targets");
        }
        let mut sorted = targets.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != targets.len() {
            bail!("stimulus targets must be unique");
        }
        Ok(Self {
            targets,
            counts,
            steps,
        })
    }

    pub fn empty(steps: usize, neuron_count: usize) -> Self {
        Self::new(Vec::new(), Vec::new(), steps, neuron_count).unwrap()
    }

    pub fn bernoulli(
        targets: Vec<u32>,
        steps: usize,
        rate_hz: f64,
        dt_ms: f64,
        seed: u64,
        neuron_count: usize,
    ) -> Result<Self> {
        if !rate_hz.is_finite() || rate_hz < 0.0 {
            bail!("stimulus rate must be finite and non-negative");
        }
        let probability = rate_hz * dt_ms / 1000.0;
        if probability > 1.0 {
            bail!("rate × timestep cannot exceed one for N=1 input");
        }
        let mut counts = Vec::with_capacity(steps.saturating_mul(targets.len()));
        for tick in 0..steps {
            for lane in 0..targets.len() {
                let key = seed
                    ^ (tick as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ (lane as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                let sample = splitmix64(key);
                let uniform = (sample >> 11) as f64 * (1.0 / (1_u64 << 53) as f64);
                counts.push(u8::from(uniform < probability));
            }
        }
        Self::new(targets, counts, steps, neuron_count)
    }

    pub fn targets(&self) -> &[u32] {
        &self.targets
    }

    pub fn counts(&self) -> &[u8] {
        &self.counts
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn events_at(&self, step: usize) -> impl Iterator<Item = (u32, u8)> + '_ {
        let start = step * self.targets.len();
        self.targets
            .iter()
            .copied()
            .zip(
                self.counts[start..start + self.targets.len()]
                    .iter()
                    .copied(),
            )
            .filter(|(_, count)| *count != 0)
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::EventSchedule;

    #[test]
    fn counter_schedule_is_deterministic() {
        let first = EventSchedule::bernoulli(vec![1, 3], 100, 150.0, 0.1, 7, 4).unwrap();
        let second = EventSchedule::bernoulli(vec![1, 3], 100, 150.0, 0.1, 7, 4).unwrap();

        assert_eq!(first, second);
        assert!(first.counts().iter().any(|&count| count != 0));
    }
}
