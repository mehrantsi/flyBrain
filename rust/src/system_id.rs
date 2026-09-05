//! Small, deterministic building blocks for system identification.
//!
//! The module deliberately keeps the model evaluator out of the data model. A
//! caller supplies an evaluator for a parameter vector and the optimizer only
//! ever gives it a validated copy of the training trials. This makes the
//! split boundary explicit and keeps the core usable on every supported Rust
//! target.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const DATASET_SCHEMA_VERSION: u32 = 1;
pub const SYSTEM_ID_SCHEMA_VERSION: u32 = DATASET_SCHEMA_VERSION;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    pub fn is_held_out(self) -> bool {
        !matches!(self, Self::Train)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlKind {
    #[default]
    None,
    ReferenceConnectome,
    NoBrain,
    DegreeMatchedShuffle,
    SignShuffle,
    DescendingNeuronPermutation,
    SensoryLesion,
    DescendingNeuronLesion,
    LeftRightSensorySwap,
    Baseline,
    Neutral,
    OpenLoop,
    ClosedLoop,
    Sham,
    Perturbation,
    Custom(String),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CandidatePathway {
    pub source_group: String,
    pub destination_group: String,
    pub signed_contact_count: i16,
    pub gain: f64,
}

impl CandidatePathway {
    pub fn validate(&self) -> Result<()> {
        validate_name(&self.source_group, "candidate pathway source_group")?;
        validate_name(
            &self.destination_group,
            "candidate pathway destination_group",
        )?;
        if self.source_group == self.destination_group {
            bail!("candidate pathway source and destination groups must differ")
        }
        if self.signed_contact_count == 0 {
            bail!("candidate pathway signed_contact_count must not be zero")
        }
        if !self.gain.is_finite() {
            bail!("candidate pathway gain must be finite")
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HypothesisOverlay {
    pub schema_version: u32,
    pub base_topology: TopologyIdentity,
    pub pathways: Vec<CandidatePathway>,
}

impl HypothesisOverlay {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SYSTEM_ID_SCHEMA_VERSION {
            bail!("unsupported hypothesis overlay schema version")
        }
        self.base_topology.validate()?;
        let mut pairs = BTreeSet::new();
        for pathway in &self.pathways {
            pathway.validate()?;
            if !pairs.insert((
                pathway.source_group.as_str(),
                pathway.destination_group.as_str(),
            )) {
                bail!("hypothesis overlay contains a duplicate group pair")
            }
        }
        Ok(())
    }

    pub fn l1_gain(&self) -> f64 {
        self.pathways.iter().map(|pathway| pathway.gain.abs()).sum()
    }

    pub fn fingerprint(&self) -> Result<String> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(b"flybrain-system-id-hypothesis-overlay-v1\0");
        hasher.update(self.base_topology.topology_sha256.as_bytes());
        for pathway in &self.pathways {
            hasher.update((pathway.source_group.len() as u64).to_le_bytes());
            hasher.update(pathway.source_group.as_bytes());
            hasher.update((pathway.destination_group.len() as u64).to_le_bytes());
            hasher.update(pathway.destination_group.as_bytes());
            hasher.update(pathway.signed_contact_count.to_le_bytes());
            hasher.update(pathway.gain.to_bits().to_le_bytes());
        }
        Ok(format!("{:x}", hasher.finalize()))
    }
}

impl ControlKind {
    fn validate(&self) -> Result<()> {
        if let Self::Custom(name) = self {
            validate_name(name, "control kind")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TopologyIdentity {
    pub topology_id: String,
    pub topology_sha256: String,
}

impl TopologyIdentity {
    pub fn new(topology_id: impl Into<String>, topology_sha256: impl Into<String>) -> Result<Self> {
        let identity = Self {
            topology_id: topology_id.into(),
            topology_sha256: topology_sha256.into(),
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(&self) -> Result<()> {
        validate_name(&self.topology_id, "topology_id")?;
        if self.topology_sha256.len() != 64
            || !self
                .topology_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            bail!("topology_sha256 must be a 64-character hexadecimal SHA-256 digest");
        }
        Ok(())
    }

    pub fn id(&self) -> &str {
        &self.topology_id
    }

    pub fn sha256(&self) -> &str {
        &self.topology_sha256
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MetricTarget {
    pub name: String,
    pub target: f64,
    pub scale: f64,
    pub weight: f64,
    pub huber_delta: f64,
}

impl MetricTarget {
    pub fn new(name: impl Into<String>, target: f64, scale: f64, weight: f64) -> Self {
        Self {
            name: name.into(),
            target,
            scale,
            weight,
            huber_delta: 1.0,
        }
    }

    pub fn with_huber_delta(mut self, huber_delta: f64) -> Self {
        self.huber_delta = huber_delta;
        self
    }

    pub fn validate(&self) -> Result<()> {
        validate_name(&self.name, "metric name")?;
        if !self.target.is_finite() {
            bail!("metric {} target must be finite", self.name);
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            bail!("metric {} scale must be finite and positive", self.name);
        }
        if !self.weight.is_finite() || self.weight < 0.0 {
            bail!(
                "metric {} weight must be finite and non-negative",
                self.name
            );
        }
        if !self.huber_delta.is_finite() || self.huber_delta <= 0.0 {
            bail!(
                "metric {} huber_delta must be finite and positive",
                self.name
            );
        }
        Ok(())
    }
}

pub type MetricSpec = MetricTarget;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MetricObservation {
    pub name: String,
    pub value: f64,
}

impl MetricObservation {
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_name(&self.name, "metric observation name")?;
        if !self.value.is_finite() {
            bail!("metric observation {} must be finite", self.name);
        }
        Ok(())
    }
}

pub type MetricValue = MetricObservation;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TrialRecord {
    pub trial_id: String,
    pub family_id: String,
    pub split: Split,
    #[serde(default)]
    pub control: ControlKind,
    #[serde(default)]
    pub condition: BTreeMap<String, String>,
    #[serde(default)]
    pub observations: Vec<MetricObservation>,
}

impl TrialRecord {
    pub fn new(
        trial_id: impl Into<String>,
        family_id: impl Into<String>,
        split: Split,
        control: ControlKind,
        observations: Vec<MetricObservation>,
    ) -> Self {
        Self {
            trial_id: trial_id.into(),
            family_id: family_id.into(),
            split,
            control,
            condition: BTreeMap::new(),
            observations,
        }
    }

    pub fn without_observations(
        trial_id: impl Into<String>,
        family_id: impl Into<String>,
        split: Split,
        control: ControlKind,
    ) -> Self {
        Self::new(trial_id, family_id, split, control, Vec::new())
    }

    pub fn with_condition(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.condition.insert(key.into(), value.into());
        self
    }

    pub fn validate(&self, metric_names: &BTreeSet<String>) -> Result<()> {
        validate_name(&self.trial_id, "trial_id")?;
        validate_name(&self.family_id, "family_id")?;
        self.control.validate()?;
        for (key, value) in &self.condition {
            validate_name(key, "trial condition key")?;
            validate_name(value, "trial condition value")?;
        }
        let mut names = BTreeSet::new();
        for observation in &self.observations {
            observation.validate()?;
            if !metric_names.contains(&observation.name) {
                bail!(
                    "trial {} contains unknown metric {}",
                    self.trial_id,
                    observation.name
                );
            }
            if !names.insert(observation.name.clone()) {
                bail!(
                    "trial {} contains duplicate metric observation {}",
                    self.trial_id,
                    observation.name
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Dataset {
    pub schema_version: u32,
    pub dataset_id: String,
    pub topology: TopologyIdentity,
    #[serde(default)]
    pub metrics: Vec<MetricTarget>,
    pub trials: Vec<TrialRecord>,
}

impl Dataset {
    pub fn new(
        dataset_id: impl Into<String>,
        topology: TopologyIdentity,
        metrics: Vec<MetricTarget>,
        trials: Vec<TrialRecord>,
    ) -> Self {
        Self {
            schema_version: DATASET_SCHEMA_VERSION,
            dataset_id: dataset_id.into(),
            topology,
            metrics,
            trials,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != DATASET_SCHEMA_VERSION {
            bail!(
                "unsupported system-identification schema version {}; expected {}",
                self.schema_version,
                DATASET_SCHEMA_VERSION
            );
        }
        validate_name(&self.dataset_id, "dataset_id")?;
        self.topology.validate()?;

        let mut metric_names = BTreeSet::new();
        let mut total_weight = 0.0;
        for metric in &self.metrics {
            metric.validate()?;
            if !metric_names.insert(metric.name.clone()) {
                bail!("duplicate metric target {}", metric.name);
            }
            total_weight += metric.weight;
        }
        if !self.metrics.is_empty() && total_weight <= 0.0 {
            bail!("metric targets must have a positive total weight");
        }

        let mut trial_ids = BTreeSet::new();
        let mut families = BTreeMap::<String, Split>::new();
        for trial in &self.trials {
            trial.validate(&metric_names)?;
            if !trial_ids.insert(trial.trial_id.clone()) {
                bail!("duplicate trial_id {}", trial.trial_id);
            }
            if let Some(previous_split) = families.insert(trial.family_id.clone(), trial.split)
                && previous_split != trial.split
            {
                bail!(
                    "family_id {} crosses {:?} and {:?} splits",
                    trial.family_id,
                    previous_split,
                    trial.split
                );
            }
        }
        Ok(())
    }

    pub fn train_trials(&self) -> Vec<TrialRecord> {
        self.trials
            .iter()
            .filter(|trial| trial.split == Split::Train)
            .cloned()
            .collect()
    }

    pub fn held_out_trials(&self, split: Split) -> Vec<TrialRecord> {
        self.trials
            .iter()
            .filter(|trial| trial.split == split)
            .cloned()
            .collect()
    }

    pub fn trials_for_split(&self, split: Split) -> Vec<TrialRecord> {
        self.held_out_trials(split)
    }

    pub fn control_reports(&self, split: Split) -> Result<Vec<ControlReport>> {
        self.validate()?;
        let mut grouped = BTreeMap::<ControlKind, Vec<(&TrialRecord, f64)>>::new();
        for trial in self.trials.iter().filter(|trial| trial.split == split) {
            let score = if self.metrics.is_empty() {
                0.0
            } else {
                score_metrics(&self.metrics, &trial.observations)?.score
            };
            grouped
                .entry(trial.control.clone())
                .or_default()
                .push((trial, score));
        }
        Ok(grouped
            .into_iter()
            .map(|(control, trials)| ControlReport {
                split,
                control,
                trial_ids: trials
                    .iter()
                    .map(|(trial, _)| trial.trial_id.clone())
                    .collect(),
                trial_count: trials.len(),
                mean_score: trials.iter().map(|(_, score)| score).sum::<f64>()
                    / trials.len() as f64,
            })
            .collect())
    }

    pub fn control_report(&self, split: Split) -> Result<Vec<ControlReport>> {
        self.control_reports(split)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ControlReport {
    pub split: Split,
    pub control: ControlKind,
    pub trial_ids: Vec<String>,
    pub trial_count: usize,
    pub mean_score: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MetricScore {
    pub name: String,
    pub target: f64,
    pub observed: f64,
    pub residual: f64,
    pub normalized_residual: f64,
    pub loss: f64,
    pub weight: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ScoreReport {
    pub score: f64,
    pub metrics: Vec<MetricScore>,
}

impl ScoreReport {
    pub fn loss(&self) -> f64 {
        self.score
    }
}

pub fn huber_loss(residual: f64, delta: f64) -> Result<f64> {
    if !residual.is_finite() {
        bail!("Huber residual must be finite");
    }
    if !delta.is_finite() || delta <= 0.0 {
        bail!("Huber delta must be finite and positive");
    }
    let magnitude = residual.abs();
    Ok(if magnitude <= delta {
        0.5 * magnitude * magnitude
    } else {
        delta * (magnitude - 0.5 * delta)
    })
}

pub fn score_metrics(
    targets: &[MetricTarget],
    observations: &[MetricObservation],
) -> Result<ScoreReport> {
    let mut target_by_name = BTreeMap::new();
    let mut total_weight = 0.0;
    for target in targets {
        target.validate()?;
        if target_by_name
            .insert(target.name.as_str(), target)
            .is_some()
        {
            bail!("duplicate metric target {}", target.name);
        }
        total_weight += target.weight;
    }
    if targets.is_empty() {
        if observations.is_empty() {
            return Ok(ScoreReport {
                score: 0.0,
                metrics: Vec::new(),
            });
        }
        bail!("observations were supplied without metric targets");
    }
    if total_weight <= 0.0 {
        bail!("metric targets must have a positive total weight");
    }

    let mut observed_by_name = BTreeMap::new();
    for observation in observations {
        observation.validate()?;
        if observed_by_name
            .insert(observation.name.as_str(), observation.value)
            .is_some()
        {
            bail!("duplicate metric observation {}", observation.name);
        }
    }
    if observed_by_name.len() != target_by_name.len() {
        bail!(
            "expected {} metric observations, got {}",
            target_by_name.len(),
            observed_by_name.len()
        );
    }

    let mut metrics = Vec::with_capacity(targets.len());
    let mut weighted_loss = 0.0;
    for target in targets {
        let observed = observed_by_name
            .get(target.name.as_str())
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing metric observation {}", target.name))?;
        let residual = observed - target.target;
        let normalized_residual = residual / target.scale;
        let loss = huber_loss(normalized_residual, target.huber_delta)?;
        weighted_loss += target.weight * loss;
        metrics.push(MetricScore {
            name: target.name.clone(),
            target: target.target,
            observed,
            residual,
            normalized_residual,
            loss,
            weight: target.weight,
        });
    }
    Ok(ScoreReport {
        score: weighted_loss / total_weight,
        metrics,
    })
}

pub fn normalized_weighted_score(
    targets: &[MetricTarget],
    observations: &[MetricObservation],
) -> Result<f64> {
    Ok(score_metrics(targets, observations)?.score)
}

pub fn robust_weighted_score(
    targets: &[MetricTarget],
    observations: &[MetricObservation],
) -> Result<f64> {
    normalized_weighted_score(targets, observations)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NamedParameter {
    pub name: String,
    pub lower: f64,
    pub upper: f64,
    pub value: f64,
}

impl NamedParameter {
    pub fn new(name: impl Into<String>, lower: f64, upper: f64, value: f64) -> Self {
        Self {
            name: name.into(),
            lower,
            upper,
            value,
        }
    }

    pub fn bounded(name: impl Into<String>, lower: f64, upper: f64, value: f64) -> Self {
        Self::new(name, lower, upper, value)
    }

    pub fn validate(&self) -> Result<()> {
        validate_name(&self.name, "parameter name")?;
        if !self.lower.is_finite()
            || !self.upper.is_finite()
            || !self.value.is_finite()
            || self.lower > self.upper
        {
            bail!("parameter {} has invalid finite bounds/value", self.name);
        }
        if self.value < self.lower || self.value > self.upper {
            bail!(
                "parameter {} value {} is outside [{}, {}]",
                self.name,
                self.value,
                self.lower,
                self.upper
            );
        }
        Ok(())
    }
}

pub type BoundedParameter = NamedParameter;
pub type Parameter = NamedParameter;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ParameterVector {
    pub parameters: Vec<NamedParameter>,
}

impl ParameterVector {
    pub fn new(parameters: Vec<NamedParameter>) -> Result<Self> {
        let vector = Self { parameters };
        vector.validate()?;
        Ok(vector)
    }

    pub fn single(name: impl Into<String>, lower: f64, upper: f64, value: f64) -> Result<Self> {
        Self::new(vec![NamedParameter::new(name, lower, upper, value)])
    }

    pub fn validate(&self) -> Result<()> {
        if self.parameters.is_empty() {
            bail!("parameter vector must not be empty");
        }
        let mut names = BTreeSet::new();
        for parameter in &self.parameters {
            parameter.validate()?;
            if !names.insert(parameter.name.clone()) {
                bail!("duplicate parameter name {}", parameter.name);
            }
        }
        Ok(())
    }

    pub fn dimension(&self) -> usize {
        self.parameters.len()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.parameters
            .iter()
            .map(|parameter| parameter.name.as_str())
    }

    pub fn values(&self) -> Vec<f64> {
        self.parameters
            .iter()
            .map(|parameter| parameter.value)
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        self.parameters
            .iter()
            .find(|parameter| parameter.name == name)
            .map(|parameter| parameter.value)
    }

    pub fn with_values(&self, values: &[f64]) -> Result<Self> {
        if values.len() != self.parameters.len() {
            bail!(
                "expected {} parameter values, got {}",
                self.parameters.len(),
                values.len()
            );
        }
        let mut result = self.clone();
        for (parameter, value) in result.parameters.iter_mut().zip(values) {
            parameter.value = *value;
        }
        result.clamp_in_place();
        result.validate()?;
        Ok(result)
    }

    pub fn set(&mut self, name: &str, value: f64) -> Result<()> {
        let parameter = self
            .parameters
            .iter_mut()
            .find(|parameter| parameter.name == name)
            .ok_or_else(|| anyhow::anyhow!("unknown parameter {name}"))?;
        parameter.value = value;
        parameter.validate()
    }

    pub fn clamped(&self) -> Result<Self> {
        let mut result = self.clone();
        result.clamp_in_place();
        result.validate()?;
        Ok(result)
    }

    fn clamp_in_place(&mut self) {
        for parameter in &mut self.parameters {
            parameter.value = parameter.value.clamp(parameter.lower, parameter.upper);
        }
    }

    fn update_normalized(&self, gradient: &[f64], step: f64) -> Result<Self> {
        if gradient.len() != self.parameters.len() {
            bail!("gradient dimension does not match parameter vector");
        }
        let values = self
            .parameters
            .iter()
            .zip(gradient)
            .map(|(parameter, derivative)| {
                parameter.value - step * derivative * (parameter.upper - parameter.lower)
            })
            .collect::<Vec<_>>();
        self.with_values(&values)
    }

    fn perturb_normalized(&self, signs: &[f64], amount: f64) -> Result<Self> {
        if signs.len() != self.parameters.len() {
            bail!("perturbation dimension does not match parameter vector");
        }
        let values = self
            .parameters
            .iter()
            .zip(signs)
            .map(|(parameter, sign)| {
                parameter.value + amount * sign * (parameter.upper - parameter.lower)
            })
            .collect::<Vec<_>>();
        self.with_values(&values)
    }

    pub fn fingerprint(&self) -> String {
        candidate_fingerprint(self)
    }
}

pub type NamedParameterVector = ParameterVector;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OptimizerConfig {
    pub iterations: usize,
    pub learning_rate: f64,
    pub perturbation: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub top_k: usize,
    pub checkpoint_every: usize,
    pub seed: u64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            iterations: 80,
            learning_rate: 0.02,
            perturbation: 0.1,
            alpha: 0.602,
            gamma: 0.101,
            top_k: 5,
            checkpoint_every: 10,
            seed: 0x5359_5354_454d_4944,
        }
    }
}

impl OptimizerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.iterations == 0 {
            bail!("optimizer iterations must be positive");
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            bail!("optimizer learning_rate must be finite and positive");
        }
        if !self.perturbation.is_finite() || self.perturbation <= 0.0 {
            bail!("optimizer perturbation must be finite and positive");
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            bail!("optimizer alpha must be finite and positive");
        }
        if !self.gamma.is_finite() || self.gamma <= 0.0 {
            bail!("optimizer gamma must be finite and positive");
        }
        if self.top_k == 0 {
            bail!("optimizer top_k must be positive");
        }
        if self.checkpoint_every == 0 {
            bail!("optimizer checkpoint_every must be positive");
        }
        Ok(())
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn with_perturbation(mut self, perturbation: f64) -> Self {
        self.perturbation = perturbation;
        self
    }

    pub fn with_checkpoint_every(mut self, checkpoint_every: usize) -> Self {
        self.checkpoint_every = checkpoint_every;
        self
    }
}

pub type SpsaConfig = OptimizerConfig;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Candidate {
    pub iteration: usize,
    pub parameters: ParameterVector,
    pub train_score: f64,
    pub fingerprint: String,
}

pub type EnsembleCandidate = Candidate;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OptimizerCheckpoint {
    pub iteration: usize,
    pub parameters: ParameterVector,
    pub train_score: f64,
    pub fingerprint: String,
}

pub type Checkpoint = OptimizerCheckpoint;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OptimizationResult {
    pub method: String,
    pub train_trial_ids: Vec<String>,
    pub evaluations: usize,
    pub best: Candidate,
    pub candidates: Vec<Candidate>,
    pub checkpoints: Vec<OptimizerCheckpoint>,
}

impl OptimizationResult {
    pub fn top_candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    pub fn best_parameters(&self) -> &ParameterVector {
        &self.best.parameters
    }
}

pub trait TrainObjective {
    fn evaluate(
        &mut self,
        parameters: &ParameterVector,
        train_trials: &[TrialRecord],
    ) -> Result<f64>;
}

impl<F> TrainObjective for F
where
    F: FnMut(&ParameterVector, &[TrialRecord]) -> Result<f64>,
{
    fn evaluate(
        &mut self,
        parameters: &ParameterVector,
        train_trials: &[TrialRecord],
    ) -> Result<f64> {
        self(parameters, train_trials)
    }
}

pub fn optimize_train<O>(
    dataset: &Dataset,
    initial: &ParameterVector,
    config: OptimizerConfig,
    mut objective: O,
) -> Result<OptimizationResult>
where
    O: TrainObjective,
{
    dataset.validate()?;
    initial.validate()?;
    config.validate()?;
    let train_trials = dataset.train_trials();
    if train_trials.is_empty() {
        bail!("system-identification optimizer requires at least one train trial");
    }

    let mut evaluations = 0usize;
    let mut evaluate = |parameters: &ParameterVector| -> Result<f64> {
        let score = objective.evaluate(parameters, &train_trials)?;
        if !score.is_finite() {
            bail!("optimizer objective returned a non-finite score");
        }
        evaluations = evaluations
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("optimizer evaluation counter overflow"))?;
        Ok(score)
    };

    let mut current = initial.clamped()?;
    let mut current_score = evaluate(&current)?;
    let mut checkpoints = vec![OptimizerCheckpoint {
        iteration: 0,
        parameters: current.clone(),
        train_score: current_score,
        fingerprint: current.fingerprint(),
    }];
    let mut history = vec![Candidate {
        iteration: 0,
        parameters: current.clone(),
        train_score: current_score,
        fingerprint: current.fingerprint(),
    }];

    for iteration in 0..config.iterations {
        let k = (iteration + 1) as f64;
        let learning_rate = config.learning_rate / k.powf(config.alpha);
        let perturbation = config.perturbation / k.powf(config.gamma);
        let signs = (0..current.dimension())
            .map(|dimension| {
                if splitmix64(config.seed ^ ((iteration as u64) << 32) ^ dimension as u64) & 1 == 0
                {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect::<Vec<_>>();
        let plus = current.perturb_normalized(&signs, perturbation)?;
        let minus = current.perturb_normalized(
            &signs.iter().map(|sign| -*sign).collect::<Vec<_>>(),
            perturbation,
        )?;
        let plus_score = evaluate(&plus)?;
        let minus_score = evaluate(&minus)?;
        let denominator = 2.0 * perturbation;
        let gradient = signs
            .iter()
            .map(|sign| (plus_score - minus_score) / (denominator * sign))
            .collect::<Vec<_>>();
        let next = current.update_normalized(&gradient, learning_rate)?;
        let next_score = evaluate(&next)?;
        current = next;
        current_score = next_score;
        let completed_iteration = iteration + 1;
        history.push(Candidate {
            iteration: completed_iteration,
            parameters: current.clone(),
            train_score: current_score,
            fingerprint: current.fingerprint(),
        });
        if completed_iteration % config.checkpoint_every == 0
            || completed_iteration == config.iterations
        {
            checkpoints.push(OptimizerCheckpoint {
                iteration: completed_iteration,
                parameters: current.clone(),
                train_score: current_score,
                fingerprint: current.fingerprint(),
            });
        }
    }

    history.sort_by(|left, right| {
        left.train_score
            .total_cmp(&right.train_score)
            .then_with(|| left.fingerprint.cmp(&right.fingerprint))
            .then_with(|| left.iteration.cmp(&right.iteration))
    });
    let mut seen = BTreeSet::new();
    history.retain(|candidate| seen.insert(candidate.fingerprint.clone()));
    history.truncate(config.top_k);
    let best = history
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("optimizer produced no candidates"))?;
    Ok(OptimizationResult {
        method: "spsa-v1".to_string(),
        train_trial_ids: train_trials
            .iter()
            .map(|trial| trial.trial_id.clone())
            .collect(),
        evaluations,
        best,
        candidates: history,
        checkpoints,
    })
}

pub fn optimize<O>(
    dataset: &Dataset,
    initial: &ParameterVector,
    config: OptimizerConfig,
    objective: O,
) -> Result<OptimizationResult>
where
    O: TrainObjective,
{
    optimize_train(dataset, initial, config, objective)
}

pub trait ParameterObjective {
    fn evaluate_parameter(&mut self, parameters: &ParameterVector) -> Result<f64>;
}

impl<F> ParameterObjective for F
where
    F: FnMut(&ParameterVector) -> Result<f64>,
{
    fn evaluate_parameter(&mut self, parameters: &ParameterVector) -> Result<f64> {
        self(parameters)
    }
}

pub fn optimize_train_parameters<O>(
    dataset: &Dataset,
    initial: &ParameterVector,
    config: OptimizerConfig,
    mut objective: O,
) -> Result<OptimizationResult>
where
    O: ParameterObjective,
{
    optimize_train(
        dataset,
        initial,
        config,
        move |parameters: &ParameterVector, _train: &[TrialRecord]| {
            objective.evaluate_parameter(parameters)
        },
    )
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HeldOutReport {
    pub split: Split,
    pub trial_ids: Vec<String>,
    pub candidate_fingerprint: String,
    pub score: f64,
    pub control_reports: Vec<ControlReport>,
}

pub fn evaluate_held_out<O>(
    dataset: &Dataset,
    parameters: &ParameterVector,
    split: Split,
    mut objective: O,
) -> Result<HeldOutReport>
where
    O: FnMut(&ParameterVector, &[TrialRecord]) -> Result<f64>,
{
    dataset.validate()?;
    parameters.validate()?;
    if !split.is_held_out() {
        bail!("held-out evaluation requires validation or test split");
    }
    let trials = dataset.held_out_trials(split);
    if trials.is_empty() {
        bail!("held-out split {:?} has no trials", split);
    }
    let score = objective(parameters, &trials)?;
    if !score.is_finite() {
        bail!("held-out objective returned a non-finite score");
    }
    Ok(HeldOutReport {
        split,
        trial_ids: trials.iter().map(|trial| trial.trial_id.clone()).collect(),
        candidate_fingerprint: parameters.fingerprint(),
        score,
        control_reports: dataset.control_reports(split)?,
    })
}

pub fn evaluate_holdout<O>(
    dataset: &Dataset,
    parameters: &ParameterVector,
    split: Split,
    objective: O,
) -> Result<HeldOutReport>
where
    O: FnMut(&ParameterVector, &[TrialRecord]) -> Result<f64>,
{
    evaluate_held_out(dataset, parameters, split, objective)
}

pub fn evaluate_test<O>(
    dataset: &Dataset,
    parameters: &ParameterVector,
    objective: O,
) -> Result<HeldOutReport>
where
    O: FnMut(&ParameterVector, &[TrialRecord]) -> Result<f64>,
{
    evaluate_held_out(dataset, parameters, Split::Test, objective)
}

pub fn candidate_fingerprint(parameters: &ParameterVector) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"flybrain-system-id-candidate-v1\0");
    hasher.update((parameters.parameters.len() as u64).to_le_bytes());
    for parameter in &parameters.parameters {
        hasher.update((parameter.name.len() as u64).to_le_bytes());
        hasher.update(parameter.name.as_bytes());
        hasher.update(parameter.lower.to_bits().to_le_bytes());
        hasher.update(parameter.upper.to_bits().to_le_bytes());
        hasher.update(parameter.value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn validate_name(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        bail!("{field} must be non-empty and contain no control characters");
    }
    Ok(())
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology() -> TopologyIdentity {
        TopologyIdentity::new("toy-topology-v1", "a".repeat(64)).unwrap()
    }

    fn metric(name: &str) -> MetricTarget {
        MetricTarget::new(name, 0.0, 1.0, 1.0)
    }

    fn trial(id: &str, family: &str, split: Split) -> TrialRecord {
        TrialRecord::new(
            id,
            family,
            split,
            ControlKind::None,
            vec![MetricObservation::new("loss", 0.0)],
        )
    }

    fn dataset(trials: Vec<TrialRecord>) -> Dataset {
        Dataset::new("toy-dataset-v1", topology(), vec![metric("loss")], trials)
    }

    fn parameters(x: f64, y: f64) -> ParameterVector {
        ParameterVector::new(vec![
            NamedParameter::new("x", -5.0, 5.0, x),
            NamedParameter::new("y", -5.0, 5.0, y),
        ])
        .unwrap()
    }

    #[test]
    fn rejects_family_split_leakage_and_duplicate_trial_ids() {
        let leakage = dataset(vec![
            trial("train-1", "family-a", Split::Train),
            trial("valid-1", "family-a", Split::Validation),
        ]);
        assert!(leakage.validate().is_err());

        let duplicate = dataset(vec![
            trial("same", "family-a", Split::Train),
            trial("same", "family-b", Split::Train),
        ]);
        assert!(duplicate.validate().is_err());
    }

    #[test]
    fn zero_score_and_metric_scaling_are_exact() {
        let targets = vec![MetricTarget::new("distance", 10.0, 2.0, 1.0)];
        let zero = score_metrics(&targets, &[MetricObservation::new("distance", 10.0)]).unwrap();
        assert_eq!(zero.score, 0.0);
        let scaled = score_metrics(&targets, &[MetricObservation::new("distance", 12.0)]).unwrap();
        assert!((scaled.score - 0.5).abs() < 1e-12);
    }

    #[test]
    fn huber_loss_is_quadratic_then_linear() {
        assert_eq!(huber_loss(0.5, 1.0).unwrap(), 0.125);
        assert_eq!(huber_loss(3.0, 1.0).unwrap(), 2.5);
    }

    #[test]
    fn parameter_vector_is_bounded_and_named() {
        let vector = parameters(0.0, 1.0);
        assert_eq!(vector.get("x"), Some(0.0));
        assert!(vector.with_values(&[10.0, -10.0]).is_ok());
        let clamped = vector.with_values(&[10.0, -10.0]).unwrap();
        assert_eq!(clamped.values(), vec![5.0, -5.0]);
    }

    #[test]
    fn optimizer_is_deterministic_recovers_quadratic_and_uses_train_only() {
        let data = dataset(vec![
            trial("train-1", "family-train", Split::Train),
            trial("validation-1", "family-validation", Split::Validation),
            trial("test-1", "family-test", Split::Test),
        ]);
        let objective = |candidate: &ParameterVector, trials: &[TrialRecord]| {
            assert!(trials.iter().all(|trial| trial.split == Split::Train));
            let x = candidate.get("x").unwrap();
            let y = candidate.get("y").unwrap();
            Ok((x - 1.25).powi(2) + (y + 0.75).powi(2))
        };
        let config = OptimizerConfig::default()
            .with_iterations(180)
            .with_top_k(8)
            .with_checkpoint_every(30)
            .with_seed(42);
        let first =
            optimize_train(&data, &parameters(0.0, 0.0), config.clone(), objective).unwrap();
        let second = optimize_train(
            &data,
            &parameters(0.0, 0.0),
            config,
            |candidate: &ParameterVector, trials: &[TrialRecord]| {
                assert!(trials.iter().all(|trial| trial.split == Split::Train));
                let x = candidate.get("x").unwrap();
                let y = candidate.get("y").unwrap();
                Ok((x - 1.25).powi(2) + (y + 0.75).powi(2))
            },
        )
        .unwrap();
        assert_eq!(first, second);
        assert!((first.best.parameters.get("x").unwrap() - 1.25).abs() < 0.35);
        assert!((first.best.parameters.get("y").unwrap() + 0.75).abs() < 0.35);
        assert_eq!(first.train_trial_ids, vec!["train-1"]);
        assert!(!first.checkpoints.is_empty());
    }

    #[test]
    fn held_out_evaluation_does_not_refit_or_pass_train() {
        let data = dataset(vec![
            trial("train-1", "family-train", Split::Train),
            trial("test-1", "family-test", Split::Test),
        ]);
        let candidate = parameters(0.0, 0.0);
        let report = evaluate_test(&data, &candidate, |_, trials| {
            assert_eq!(
                trials
                    .iter()
                    .map(|trial| trial.trial_id.as_str())
                    .collect::<Vec<_>>(),
                ["test-1"]
            );
            Ok(0.25)
        })
        .unwrap();
        assert_eq!(report.score, 0.25);
        assert_eq!(report.candidate_fingerprint, candidate.fingerprint());
    }

    #[test]
    fn candidate_fingerprint_is_deterministic_and_serde_round_trips() {
        let candidate = parameters(1.5, -2.25);
        let first = candidate_fingerprint(&candidate);
        let second = candidate_fingerprint(&candidate);
        assert_eq!(first, second);
        let data = dataset(vec![trial("train-1", "family-train", Split::Train)]);
        let json = serde_json::to_string(&data).unwrap();
        let decoded: Dataset = serde_json::from_str(&json).unwrap();
        assert_eq!(data, decoded);
        let vector_json = serde_json::to_string(&candidate).unwrap();
        let decoded_vector: ParameterVector = serde_json::from_str(&vector_json).unwrap();
        assert_eq!(candidate, decoded_vector);
    }

    #[test]
    fn hypothesis_overlay_is_bound_to_topology_and_rejects_duplicate_pairs() {
        let pathway = CandidatePathway {
            source_group: "sensory_olfactory_left".to_string(),
            destination_group: "descending_flight_left".to_string(),
            signed_contact_count: 2,
            gain: 0.1,
        };
        let overlay = HypothesisOverlay {
            schema_version: SYSTEM_ID_SCHEMA_VERSION,
            base_topology: topology(),
            pathways: vec![pathway.clone()],
        };
        assert!(overlay.validate().is_ok());
        assert_eq!(overlay.l1_gain(), 0.1);
        assert_eq!(overlay.fingerprint().unwrap().len(), 64);
        let duplicate = HypothesisOverlay {
            pathways: vec![pathway.clone(), pathway],
            ..overlay
        };
        assert!(duplicate.validate().is_err());
    }
}
