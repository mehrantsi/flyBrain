use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::system_id::{
    ControlKind, Dataset, MetricObservation, MetricTarget, Split, TopologyIdentity, TrialRecord,
};

pub const FLIGHT_TARGET_SCHEMA: &str = "flybrain.flight-system-id-targets";
pub const FLIGHT_TARGET_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct FlightTargetAsset {
    pub schema: String,
    pub schema_version: u32,
    pub generator: String,
    pub dataset: FlightDatasetSummary,
    pub source: FlightTargetSource,
    pub split: FlightSplitSummary,
    pub pairing: FlightPairing,
    pub trajectories: Vec<FlightTrajectoryTarget>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct FlightDatasetSummary {
    pub pair_count: usize,
    pub trajectory_count: usize,
    pub total_samples: usize,
    pub timestep_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct FlightTargetSource {
    pub bytes: usize,
    pub commit: String,
    pub license: String,
    pub path: String,
    pub repository: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct FlightSplitSummary {
    pub pair_counts: SplitCounts,
    pub trajectory_counts: SplitCounts,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub struct SplitCounts {
    pub train: usize,
    pub validation: usize,
    pub test: usize,
}

impl SplitCounts {
    fn get(self, split: Split) -> usize {
        match split {
            Split::Train => self.train,
            Split::Validation => self.validation,
            Split::Test => self.test,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct FlightPairing {
    pub families: Vec<String>,
    pub pairs: Vec<FlightPair>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct FlightPair {
    pub family: String,
    pub original_index: usize,
    pub reflected_index: usize,
    pub pair_id: String,
    pub split: String,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct ScalarSummary {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
}

impl ScalarSummary {
    fn validate(self, name: &str) -> Result<()> {
        if [self.min, self.max, self.mean, self.std]
            .into_iter()
            .any(|value| !value.is_finite())
            || self.min > self.max
            || self.mean < self.min - 1e-6
            || self.mean > self.max + 1e-6
            || self.std < 0.0
        {
            bail!("flight target {name} summary is invalid")
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct PostureSummary {
    pub roll: ScalarSummary,
    pub pitch: ScalarSummary,
    pub yaw_unwrapped: ScalarSummary,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct FlightTrajectoryTarget {
    pub trajectory_index: usize,
    pub trajectory_type: String,
    pub pair_id: String,
    pub reflection: String,
    pub split: String,
    pub sample_count: usize,
    pub duration_seconds: f64,
    pub heading_change_rad: f64,
    pub angular_speed_rad_s: ScalarSummary,
    pub planar_speed_cm_s: ScalarSummary,
    pub forward_speed_cm_s: ScalarSummary,
    pub lateral_speed_cm_s: ScalarSummary,
    pub vertical_speed_cm_s: ScalarSummary,
    pub turn_rate_rad_s: ScalarSummary,
    pub posture_rad: PostureSummary,
}

impl FlightTargetAsset {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("reading flight target asset {}", path.display()))?;
        let asset: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing flight target asset {}", path.display()))?;
        asset.validate()?;
        Ok(asset)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != FLIGHT_TARGET_SCHEMA
            || self.schema_version != FLIGHT_TARGET_SCHEMA_VERSION
        {
            bail!("unsupported flight target asset schema")
        }
        if self.generator.is_empty()
            || self.source.bytes == 0
            || self.source.commit.is_empty()
            || self.source.license.is_empty()
            || self.source.path.is_empty()
            || self.source.repository.is_empty()
            || !valid_sha256(&self.source.sha256)
            || !self.dataset.timestep_seconds.is_finite()
            || self.dataset.timestep_seconds <= 0.0
        {
            bail!("flight target provenance is invalid")
        }
        if self.dataset.trajectory_count != self.trajectories.len()
            || self.dataset.pair_count != self.pairing.pairs.len()
            || self.dataset.trajectory_count != self.dataset.pair_count * 2
            || self.dataset.total_samples
                != self
                    .trajectories
                    .iter()
                    .map(|trajectory| trajectory.sample_count)
                    .sum::<usize>()
        {
            bail!("flight target dataset counts do not match its records")
        }

        let mut pair_ids = BTreeSet::new();
        let mut trajectory_indices = BTreeSet::new();
        let by_index = self
            .trajectories
            .iter()
            .map(|trajectory| (trajectory.trajectory_index, trajectory))
            .collect::<BTreeMap<_, _>>();
        if by_index.len() != self.trajectories.len() {
            bail!("flight target trajectory indices must be unique")
        }
        let family_names = self.pairing.families.iter().collect::<BTreeSet<_>>();
        let mut actual_pair_counts = [0_usize; 3];
        let mut actual_trajectory_counts = [0_usize; 3];
        for pair in &self.pairing.pairs {
            if pair.pair_id.is_empty()
                || !pair_ids.insert(pair.pair_id.as_str())
                || !family_names.contains(&pair.family)
                || pair.original_index == pair.reflected_index
            {
                bail!("flight target pair metadata is invalid")
            }
            let split = parse_split(&pair.split)?;
            actual_pair_counts[split_index(split)] += 1;
            let original = by_index
                .get(&pair.original_index)
                .with_context(|| format!("pair {} has no original trajectory", pair.pair_id))?;
            let reflected = by_index
                .get(&pair.reflected_index)
                .with_context(|| format!("pair {} has no reflected trajectory", pair.pair_id))?;
            validate_pair_member(original, pair, "original", split)?;
            validate_pair_member(reflected, pair, "reflected", split)?;
        }
        for trajectory in &self.trajectories {
            validate_trajectory(trajectory, self.dataset.timestep_seconds)?;
            if !pair_ids.contains(trajectory.pair_id.as_str())
                || !trajectory_indices.insert(trajectory.trajectory_index)
            {
                bail!("flight trajectory does not resolve to one unique pair")
            }
            actual_trajectory_counts[split_index(parse_split(&trajectory.split)?)] += 1;
        }
        for split in [Split::Train, Split::Validation, Split::Test] {
            if actual_pair_counts[split_index(split)] != self.split.pair_counts.get(split)
                || actual_trajectory_counts[split_index(split)]
                    != self.split.trajectory_counts.get(split)
            {
                bail!("flight target split counts do not match the records")
            }
        }
        Ok(())
    }

    pub fn to_system_id_dataset(&self) -> Result<Dataset> {
        self.validate()?;
        let train = self
            .trajectories
            .iter()
            .filter(|trajectory| trajectory.split == "train")
            .collect::<Vec<_>>();
        let metric_definitions = metric_definitions();
        let metrics = metric_definitions
            .iter()
            .map(|definition| {
                let values = train
                    .iter()
                    .map(|trajectory| {
                        (definition.value(trajectory), trajectory.sample_count as f64)
                    })
                    .collect::<Vec<_>>();
                let target = weighted_mean(&values);
                let variance = values
                    .iter()
                    .map(|(value, weight)| weight * (value - target).powi(2))
                    .sum::<f64>()
                    / values.iter().map(|(_, weight)| weight).sum::<f64>();
                MetricTarget::new(
                    definition.name,
                    target,
                    variance.sqrt().max(definition.minimum_scale),
                    definition.weight,
                )
            })
            .collect::<Vec<_>>();
        let pair_family = self
            .pairing
            .pairs
            .iter()
            .map(|pair| (pair.pair_id.as_str(), pair.family.as_str()))
            .collect::<BTreeMap<_, _>>();
        let trials = self
            .trajectories
            .iter()
            .map(|trajectory| {
                let family = pair_family[trajectory.pair_id.as_str()];
                let observations = metric_definitions
                    .iter()
                    .map(|definition| {
                        MetricObservation::new(definition.name, definition.value(trajectory))
                    })
                    .collect();
                let trial = TrialRecord::new(
                    format!("trajectory-{:03}", trajectory.trajectory_index),
                    trajectory.pair_id.clone(),
                    parse_split(&trajectory.split)?,
                    ControlKind::ReferenceConnectome,
                    observations,
                )
                .with_condition("family", family)
                .with_condition("reflection", &trajectory.reflection);
                Ok(trial)
            })
            .collect::<Result<Vec<_>>>()?;
        let dataset = Dataset::new(
            "flybody-measured-flight-v1",
            TopologyIdentity::new("flybody-measured-body-flight", &self.source.sha256)?,
            metrics,
            trials,
        );
        dataset.validate()?;
        Ok(dataset)
    }
}

struct MetricDefinition {
    name: &'static str,
    minimum_scale: f64,
    weight: f64,
    value: fn(&FlightTrajectoryTarget) -> f64,
}

impl MetricDefinition {
    fn value(&self, trajectory: &FlightTrajectoryTarget) -> f64 {
        (self.value)(trajectory)
    }
}

fn metric_definitions() -> [MetricDefinition; 8] {
    [
        MetricDefinition {
            name: "pitch_mean_rad",
            minimum_scale: 0.05,
            weight: 1.5,
            value: |trajectory| trajectory.posture_rad.pitch.mean,
        },
        MetricDefinition {
            name: "planar_speed_mean_mm_s",
            minimum_scale: 20.0,
            weight: 1.5,
            value: |trajectory| trajectory.planar_speed_cm_s.mean * 10.0,
        },
        MetricDefinition {
            name: "forward_speed_mean_mm_s",
            minimum_scale: 20.0,
            weight: 0.5,
            value: |trajectory| trajectory.forward_speed_cm_s.mean * 10.0,
        },
        MetricDefinition {
            name: "absolute_turn_rate_mean_rad_s",
            minimum_scale: 1.0,
            weight: 1.0,
            value: |trajectory| trajectory.turn_rate_rad_s.mean.abs(),
        },
        MetricDefinition {
            name: "angular_speed_mean_rad_s",
            minimum_scale: 2.0,
            weight: 0.75,
            value: |trajectory| trajectory.angular_speed_rad_s.mean,
        },
        MetricDefinition {
            name: "vertical_speed_mean_mm_s",
            minimum_scale: 10.0,
            weight: 0.25,
            value: |trajectory| trajectory.vertical_speed_cm_s.mean * 10.0,
        },
        MetricDefinition {
            name: "heading_rate_abs_rad_s",
            minimum_scale: 1.0,
            weight: 1.5,
            value: |trajectory| trajectory.heading_change_rad.abs() / trajectory.duration_seconds,
        },
        MetricDefinition {
            name: "duration_seconds",
            minimum_scale: 0.02,
            weight: 0.0,
            value: |trajectory| trajectory.duration_seconds,
        },
    ]
}

fn validate_pair_member(
    trajectory: &FlightTrajectoryTarget,
    pair: &FlightPair,
    reflection: &str,
    split: Split,
) -> Result<()> {
    if trajectory.pair_id != pair.pair_id
        || trajectory.reflection != reflection
        || parse_split(&trajectory.split)? != split
        || !trajectory.trajectory_type.starts_with(&pair.family)
    {
        bail!(
            "pair {} contains inconsistent trajectory metadata",
            pair.pair_id
        )
    }
    Ok(())
}

fn validate_trajectory(trajectory: &FlightTrajectoryTarget, timestep_seconds: f64) -> Result<()> {
    if trajectory.trajectory_type.is_empty()
        || trajectory.pair_id.is_empty()
        || !matches!(trajectory.reflection.as_str(), "original" | "reflected")
        || trajectory.sample_count < 2
        || !trajectory.duration_seconds.is_finite()
        || trajectory.duration_seconds <= 0.0
        || !trajectory.heading_change_rad.is_finite()
        || ((trajectory.sample_count - 1) as f64 * timestep_seconds - trajectory.duration_seconds)
            .abs()
            > 1e-8
    {
        bail!(
            "flight trajectory {} is invalid",
            trajectory.trajectory_index
        )
    }
    for (name, summary) in [
        ("angular_speed", trajectory.angular_speed_rad_s),
        ("planar_speed", trajectory.planar_speed_cm_s),
        ("forward_speed", trajectory.forward_speed_cm_s),
        ("lateral_speed", trajectory.lateral_speed_cm_s),
        ("vertical_speed", trajectory.vertical_speed_cm_s),
        ("turn_rate", trajectory.turn_rate_rad_s),
        ("roll", trajectory.posture_rad.roll),
        ("pitch", trajectory.posture_rad.pitch),
        ("yaw", trajectory.posture_rad.yaw_unwrapped),
    ] {
        summary.validate(name)?;
    }
    Ok(())
}

fn weighted_mean(values: &[(f64, f64)]) -> f64 {
    values
        .iter()
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
        / values.iter().map(|(_, weight)| weight).sum::<f64>()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn parse_split(value: &str) -> Result<Split> {
    match value {
        "train" => Ok(Split::Train),
        "validation" => Ok(Split::Validation),
        "test" => Ok(Split::Test),
        _ => bail!("unknown flight target split {value}"),
    }
}

fn split_index(split: Split) -> usize {
    match split {
        Split::Train => 0,
        Split::Validation => 1,
        Split::Test => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TARGETS: &str = "assets/neuromechfly/flight_system_id_targets_v1.json";

    #[test]
    fn measured_flight_targets_validate_and_preserve_pair_splits() {
        let asset = FlightTargetAsset::load(TARGETS).unwrap();
        assert_eq!(asset.dataset.trajectory_count, 272);
        assert_eq!(asset.dataset.pair_count, 136);
        let dataset = asset.to_system_id_dataset().unwrap();
        assert_eq!(dataset.train_trials().len(), 196);
        assert_eq!(dataset.held_out_trials(Split::Validation).len(), 40);
        assert_eq!(dataset.held_out_trials(Split::Test).len(), 36);
    }

    #[test]
    fn conversion_uses_millimeters_and_training_only_statistics() {
        let asset = FlightTargetAsset::load(TARGETS).unwrap();
        let first = &asset.trajectories[0];
        let dataset = asset.to_system_id_dataset().unwrap();
        let trial = dataset
            .trials
            .iter()
            .find(|trial| trial.trial_id == "trajectory-000")
            .unwrap();
        let speed = trial
            .observations
            .iter()
            .find(|observation| observation.name == "planar_speed_mean_mm_s")
            .unwrap();
        assert_eq!(speed.value, first.planar_speed_cm_s.mean * 10.0);

        let mut changed = asset.clone();
        changed
            .trajectories
            .iter_mut()
            .filter(|trajectory| trajectory.split != "train")
            .for_each(|trajectory| trajectory.heading_change_rad += 1_000.0);
        let changed_dataset = changed.to_system_id_dataset().unwrap();
        assert_eq!(dataset.metrics, changed_dataset.metrics);
    }
}
