//! Pure-Rust embodiment interfaces for fixed-step sensing and motor decoding.
//!
//! The bridge types in this module are engineering placeholders. They do not
//! assert a biological mapping between FlyWire neurons and any body model.

use std::collections::{BTreeMap, HashSet};
use std::f64::consts::PI;

use anyhow::{Result, bail};

use crate::olfaction::{AntennaSide, FOOD_ODOR_BAND_COUNT, FoodOdorBand};

pub const LEG_COUNT: usize = 6;
pub const JOINTS_PER_LEG: usize = 3;

/// The encoder is a deterministic fractional-rate accumulator, not a claim about
/// the fly's sensory transduction or spike-generation mechanism.
pub const SENSORY_ENCODER_MODEL: &str = "engineering-deterministic-rate-accumulator-v1";
pub const SENSORY_MAPPING_POLICY: &str = "explicit-target-neuron-id-and-rate-weight";
pub const BASELINE_CPG_MODEL: &str = "engineering-six-leg-tripod-cpg-v1";
pub const BASELINE_CPG_MAPPING_POLICY: &str =
    "explicit-brain-probe-id-and-rate-weight; no-biological-claim";
pub const BASELINE_LEG_INDEX_ORDER: [&str; LEG_COUNT] = [
    "front_left",
    "middle_left",
    "rear_left",
    "front_right",
    "middle_right",
    "rear_right",
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedStepConfig {
    pub brain_dt_ms: f64,
    pub physics_dt_ms: f64,
    pub brain_steps_per_physics_step: usize,
}

impl FixedStepConfig {
    pub fn new(brain_dt_ms: f64, physics_dt_ms: f64) -> Result<Self> {
        let ratio = validate_step_durations(brain_dt_ms, physics_dt_ms)?;
        Ok(Self {
            brain_dt_ms,
            physics_dt_ms,
            brain_steps_per_physics_step: ratio,
        })
    }

    pub fn validate(self) -> Result<Self> {
        let ratio = validate_step_durations(self.brain_dt_ms, self.physics_dt_ms)?;
        if ratio != self.brain_steps_per_physics_step {
            bail!("brain_steps_per_physics_step does not match the configured durations")
        }
        Ok(self)
    }

    pub fn scheduler(self) -> Result<FixedStepScheduler> {
        FixedStepScheduler::new(self)
    }
}

fn validate_step_durations(brain_dt_ms: f64, physics_dt_ms: f64) -> Result<usize> {
    if !brain_dt_ms.is_finite() || brain_dt_ms <= 0.0 {
        bail!("brain_dt_ms must be finite and positive")
    }
    if !physics_dt_ms.is_finite() || physics_dt_ms <= 0.0 {
        bail!("physics_dt_ms must be finite and positive")
    }
    let ratio_f = physics_dt_ms / brain_dt_ms;
    let rounded = ratio_f.round();
    if rounded < 1.0
        || rounded > usize::MAX as f64
        || (ratio_f - rounded).abs() > 1e-9 * ratio_f.max(1.0)
    {
        bail!("physics_dt_ms must be an integer multiple of brain_dt_ms and not shorter")
    }
    Ok(rounded as usize)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepWindow {
    pub physics_tick: u64,
    pub brain_tick_start: u64,
    pub brain_tick_end_exclusive: u64,
}

#[derive(Clone, Debug)]
pub struct FixedStepScheduler {
    config: FixedStepConfig,
    physics_tick: u64,
    brain_tick: u64,
}

impl FixedStepScheduler {
    pub fn new(config: FixedStepConfig) -> Result<Self> {
        Ok(Self {
            config: config.validate()?,
            physics_tick: 0,
            brain_tick: 0,
        })
    }

    pub fn config(&self) -> FixedStepConfig {
        self.config
    }

    pub fn next_physics_step(&mut self) -> Result<StepWindow> {
        let brain_tick_end_exclusive = self
            .brain_tick
            .checked_add(self.config.brain_steps_per_physics_step as u64)
            .ok_or_else(|| anyhow::anyhow!("brain tick counter overflow"))?;
        let window = StepWindow {
            physics_tick: self.physics_tick,
            brain_tick_start: self.brain_tick,
            brain_tick_end_exclusive,
        };
        self.brain_tick = brain_tick_end_exclusive;
        self.physics_tick = self
            .physics_tick
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("physics tick counter overflow"))?;
        Ok(window)
    }

    pub fn physics_tick(&self) -> u64 {
        self.physics_tick
    }

    pub fn brain_tick(&self) -> u64 {
        self.brain_tick
    }

    pub fn reset(&mut self) {
        self.physics_tick = 0;
        self.brain_tick = 0;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensorySample {
    pub timestamp_ms: f64,
    pub joint_angles_rad: [[f64; JOINTS_PER_LEG]; LEG_COUNT],
    pub joint_velocities_rad_s: [[f64; JOINTS_PER_LEG]; LEG_COUNT],
    pub foot_contacts: [bool; LEG_COUNT],
    pub odor_intensity: f64,
    pub odor_left: f64,
    pub odor_right: f64,
    pub food_odor_activation: [[f64; FOOD_ODOR_BAND_COUNT]; 2],
    pub taste_valence: f64,
    pub visual_left: f64,
    pub visual_right: f64,
    pub visual_contrast_left: f64,
    pub visual_contrast_right: f64,
    pub visual_motion: [f64; 2],
    pub visual_loom: [f64; 2],
    pub angular_velocity_rad_s: [f64; 3],
    pub flight_angular_speed_rad_s: f64,
    pub flight_mechanosensory: f64,
}

impl Default for SensorySample {
    fn default() -> Self {
        Self {
            timestamp_ms: 0.0,
            joint_angles_rad: [[0.0; JOINTS_PER_LEG]; LEG_COUNT],
            joint_velocities_rad_s: [[0.0; JOINTS_PER_LEG]; LEG_COUNT],
            foot_contacts: [false; LEG_COUNT],
            odor_intensity: 0.0,
            odor_left: 0.0,
            odor_right: 0.0,
            food_odor_activation: [[0.0; FOOD_ODOR_BAND_COUNT]; 2],
            taste_valence: 0.0,
            visual_left: 0.0,
            visual_right: 0.0,
            visual_contrast_left: 0.0,
            visual_contrast_right: 0.0,
            visual_motion: [0.0; 2],
            visual_loom: [0.0; 2],
            angular_velocity_rad_s: [0.0; 3],
            flight_angular_speed_rad_s: 0.0,
            flight_mechanosensory: 0.0,
        }
    }
}

impl SensorySample {
    pub fn validate(&self) -> Result<()> {
        if !self.timestamp_ms.is_finite() {
            bail!("sensory timestamp must be finite")
        }
        for angle in self.joint_angles_rad.iter().flatten() {
            if !angle.is_finite() {
                bail!("joint angles must be finite")
            }
        }
        for velocity in self.joint_velocities_rad_s.iter().flatten() {
            if !velocity.is_finite() {
                bail!("joint velocities must be finite")
            }
        }
        if [
            self.odor_intensity,
            self.odor_left,
            self.odor_right,
            self.taste_valence,
            self.visual_left,
            self.visual_right,
            self.visual_contrast_left,
            self.visual_contrast_right,
            self.flight_angular_speed_rad_s,
            self.flight_mechanosensory,
        ]
        .iter()
        .chain(self.food_odor_activation.iter().flatten())
        .chain(self.angular_velocity_rad_s.iter())
        .chain(self.visual_motion.iter())
        .chain(self.visual_loom.iter())
        .any(|value| !value.is_finite())
            || self.flight_angular_speed_rad_s < 0.0
            || !(0.0..=1.0).contains(&self.flight_mechanosensory)
            || self
                .food_odor_activation
                .iter()
                .flatten()
                .any(|value| !(0.0..=1.0).contains(value))
        {
            bail!("sensory scalar and angular-velocity values must be finite")
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SensoryFeature {
    JointAngle {
        leg_index: usize,
        joint_index: usize,
    },
    JointVelocity {
        leg_index: usize,
        joint_index: usize,
    },
    FootContact {
        leg_index: usize,
    },
    OdorIntensity,
    OdorLeft,
    OdorRight,
    FoodOdor {
        side: AntennaSide,
        band: FoodOdorBand,
    },
    TasteValence,
    VisualLeft,
    VisualRight,
    VisualContrastLeft,
    VisualContrastRight,
    VisualMotionLeft,
    VisualMotionRight,
    VisualLoomLeft,
    VisualLoomRight,
    AngularVelocity {
        axis: usize,
    },
    FlightAngularSpeed,
    FlightMechanosensory,
}

impl SensoryFeature {
    fn validate(self) -> Result<()> {
        match self {
            Self::JointAngle {
                leg_index,
                joint_index,
            }
            | Self::JointVelocity {
                leg_index,
                joint_index,
            } => {
                if leg_index >= LEG_COUNT || joint_index >= JOINTS_PER_LEG {
                    bail!("joint sensory feature index is outside the six-leg sample")
                }
            }
            Self::FootContact { leg_index } if leg_index >= LEG_COUNT => {
                bail!("contact sensory feature index is outside the six-leg sample")
            }
            Self::AngularVelocity { axis } if axis >= 3 => {
                bail!("angular-velocity sensory feature axis is outside [0, 3)")
            }
            _ => {}
        }
        Ok(())
    }

    fn value(self, sample: &SensorySample) -> f64 {
        match self {
            Self::JointAngle {
                leg_index,
                joint_index,
            } => sample.joint_angles_rad[leg_index][joint_index],
            Self::JointVelocity {
                leg_index,
                joint_index,
            } => sample.joint_velocities_rad_s[leg_index][joint_index],
            Self::FootContact { leg_index } => {
                if sample.foot_contacts[leg_index] {
                    1.0
                } else {
                    0.0
                }
            }
            Self::OdorIntensity => sample.odor_intensity,
            Self::OdorLeft => sample.odor_left,
            Self::OdorRight => sample.odor_right,
            Self::FoodOdor { side, band } => {
                sample.food_odor_activation[side as usize][band as usize]
            }
            Self::TasteValence => sample.taste_valence,
            Self::VisualLeft => sample.visual_left,
            Self::VisualRight => sample.visual_right,
            Self::VisualContrastLeft => sample.visual_contrast_left,
            Self::VisualContrastRight => sample.visual_contrast_right,
            Self::VisualMotionLeft => sample.visual_motion[0],
            Self::VisualMotionRight => sample.visual_motion[1],
            Self::VisualLoomLeft => sample.visual_loom[0],
            Self::VisualLoomRight => sample.visual_loom[1],
            Self::AngularVelocity { axis } => sample.angular_velocity_rad_s[axis],
            Self::FlightAngularSpeed => sample.flight_angular_speed_rad_s,
            Self::FlightMechanosensory => sample.flight_mechanosensory,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensoryRateChannel {
    pub feature: SensoryFeature,
    pub target_neuron_id: u64,
    pub input_min: f64,
    pub input_max: f64,
    pub baseline_rate_hz: f64,
    pub rate_weight_hz_per_unit: f64,
}

impl SensoryRateChannel {
    pub fn new(
        feature: SensoryFeature,
        target_neuron_id: u64,
        input_min: f64,
        input_max: f64,
        baseline_rate_hz: f64,
        rate_weight_hz_per_unit: f64,
    ) -> Result<Self> {
        let channel = Self {
            feature,
            target_neuron_id,
            input_min,
            input_max,
            baseline_rate_hz,
            rate_weight_hz_per_unit,
        };
        channel.validate()?;
        Ok(channel)
    }

    fn validate(self) -> Result<()> {
        self.feature.validate()?;
        if self.target_neuron_id == 0 {
            bail!("sensory channel requires an explicit nonzero target_neuron_id")
        }
        if !self.input_min.is_finite()
            || !self.input_max.is_finite()
            || self.input_max <= self.input_min
        {
            bail!("sensory channel input range must be finite and increasing")
        }
        if !self.baseline_rate_hz.is_finite() || self.baseline_rate_hz < 0.0 {
            bail!("sensory baseline_rate_hz must be finite and non-negative")
        }
        if !self.rate_weight_hz_per_unit.is_finite() || self.rate_weight_hz_per_unit < 0.0 {
            bail!("sensory rate_weight_hz_per_unit must be finite and non-negative")
        }
        Ok(())
    }

    fn rate_hz(self, sample: &SensorySample) -> f64 {
        let normalized = ((self.feature.value(sample) - self.input_min)
            / (self.input_max - self.input_min))
            .clamp(0.0, 1.0);
        self.baseline_rate_hz + normalized * self.rate_weight_hz_per_unit
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SensoryBridgeConfig {
    pub channels: Vec<SensoryRateChannel>,
}

impl SensoryBridgeConfig {
    pub fn new(channels: Vec<SensoryRateChannel>) -> Result<Self> {
        let config = Self { channels };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let mut target_ids = HashSet::with_capacity(self.channels.len());
        for channel in &self.channels {
            channel.validate()?;
            if !target_ids.insert(channel.target_neuron_id) {
                bail!("sensory bridge target neuron IDs must be unique")
            }
        }
        Ok(())
    }

    pub fn metadata(&self) -> (&'static str, &'static str) {
        (SENSORY_ENCODER_MODEL, SENSORY_MAPPING_POLICY)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseExternalEvent {
    pub target_neuron_id: u64,
    pub count: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingSensoryEvent {
    step: u32,
    lane: u32,
    count: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseSensoryWindow<'a> {
    steps: usize,
    step_offsets: &'a [u32],
    lanes: &'a [u32],
    counts: &'a [u8],
}

impl SparseSensoryWindow<'_> {
    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn step_offsets(&self) -> &[u32] {
        self.step_offsets
    }

    pub fn lanes(&self) -> &[u32] {
        self.lanes
    }

    pub fn counts(&self) -> &[u8] {
        self.counts
    }
}

/// Deterministic rate-to-count encoder for explicitly configured external IDs.
///
/// Each channel maps its selected sample feature into a clamped rate, then
/// carries the fractional count between brain steps. The sparse output uses
/// the same `u8` event-count convention as the existing engine stimulus API.
#[derive(Clone, Debug)]
pub struct SensoryRateEncoder {
    config: SensoryBridgeConfig,
    brain_dt_ms: f64,
    phase_cycles: Vec<f64>,
    pending_events: Vec<PendingSensoryEvent>,
    window_step_offsets: Vec<u32>,
    window_lanes: Vec<u32>,
    window_counts: Vec<u8>,
    window_write_positions: Vec<usize>,
}

impl SensoryRateEncoder {
    pub fn new(config: SensoryBridgeConfig, brain_dt_ms: f64) -> Result<Self> {
        config.validate()?;
        if !brain_dt_ms.is_finite() || brain_dt_ms <= 0.0 {
            bail!("brain_dt_ms must be finite and positive")
        }
        for channel in &config.channels {
            let max_rate_hz = channel.baseline_rate_hz + channel.rate_weight_hz_per_unit;
            if max_rate_hz * brain_dt_ms / 1000.0 > f64::from(u8::MAX) {
                bail!("sensory channel emits more than u8::MAX events per brain step")
            }
        }
        Ok(Self {
            phase_cycles: config
                .channels
                .iter()
                .map(|channel| deterministic_phase(channel.target_neuron_id))
                .collect(),
            config,
            brain_dt_ms,
            pending_events: Vec::new(),
            window_step_offsets: Vec::new(),
            window_lanes: Vec::new(),
            window_counts: Vec::new(),
            window_write_positions: Vec::new(),
        })
    }

    pub fn from_scheduler(
        config: SensoryBridgeConfig,
        scheduler: &FixedStepConfig,
    ) -> Result<Self> {
        Self::new(config, scheduler.brain_dt_ms)
    }

    pub fn config(&self) -> &SensoryBridgeConfig {
        &self.config
    }

    pub fn metadata(&self) -> (&'static str, &'static str) {
        self.config.metadata()
    }

    pub fn reset(&mut self) {
        self.phase_cycles.fill(0.0);
        self.pending_events.clear();
        self.window_step_offsets.clear();
        self.window_lanes.clear();
        self.window_counts.clear();
        self.window_write_positions.clear();
    }

    pub fn encode(&mut self, sample: &SensorySample) -> Result<Vec<SparseExternalEvent>> {
        sample.validate()?;
        let dt_seconds = self.brain_dt_ms / 1000.0;
        let mut events = Vec::new();
        for (index, channel) in self.config.channels.iter().copied().enumerate() {
            let increment = channel.rate_hz(sample) * dt_seconds;
            self.phase_cycles[index] += increment;
            let count = self.phase_cycles[index].floor();
            self.phase_cycles[index] -= count;
            if count > 0.0 {
                if count > f64::from(u8::MAX) {
                    bail!("sensory channel event count exceeds u8::MAX")
                }
                events.push(SparseExternalEvent {
                    target_neuron_id: channel.target_neuron_id,
                    count: count as u8,
                });
            }
        }
        Ok(events)
    }

    pub fn encode_window(
        &mut self,
        sample: &SensorySample,
        steps: usize,
    ) -> Result<SparseSensoryWindow<'_>> {
        sample.validate()?;
        if steps > u32::MAX as usize || self.config.channels.len() > u32::MAX as usize {
            bail!("sensory window dimensions exceed u32")
        }
        let dt_seconds = self.brain_dt_ms / 1000.0;
        self.pending_events.clear();
        for lane in 0..self.config.channels.len() {
            let increment = self.config.channels[lane].rate_hz(sample) * dt_seconds;
            for step in 0..steps {
                self.phase_cycles[lane] += increment;
                let count = self.phase_cycles[lane].floor();
                self.phase_cycles[lane] -= count;
                if count > 0.0 {
                    if count > f64::from(u8::MAX) {
                        bail!("sensory channel event count exceeds u8::MAX")
                    }
                    self.pending_events.push(PendingSensoryEvent {
                        step: step as u32,
                        lane: lane as u32,
                        count: count as u8,
                    });
                }
            }
        }
        if self.pending_events.len() > u32::MAX as usize {
            bail!("sensory window event count exceeds u32")
        }

        self.window_step_offsets.clear();
        self.window_step_offsets.resize(steps + 1, 0);
        for event in &self.pending_events {
            self.window_step_offsets[event.step as usize + 1] += 1;
        }
        for step in 0..steps {
            self.window_step_offsets[step + 1] += self.window_step_offsets[step];
        }

        self.window_lanes.clear();
        self.window_lanes.resize(self.pending_events.len(), 0);
        self.window_counts.clear();
        self.window_counts.resize(self.pending_events.len(), 0);
        self.window_write_positions.clear();
        self.window_write_positions.extend(
            self.window_step_offsets[..steps]
                .iter()
                .map(|&offset| offset as usize),
        );
        for event in &self.pending_events {
            let position = &mut self.window_write_positions[event.step as usize];
            self.window_lanes[*position] = event.lane;
            self.window_counts[*position] = event.count;
            *position += 1;
        }

        Ok(SparseSensoryWindow {
            steps,
            step_offsets: &self.window_step_offsets,
            lanes: &self.window_lanes,
            counts: &self.window_counts,
        })
    }
}

fn deterministic_phase(neuron_id: u64) -> f64 {
    let mut value = neuron_id.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BrainProbeRole {
    ForwardGain,
    TurnGain,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrainProbeBinding {
    pub neuron_id: u64,
    pub role: BrainProbeRole,
    pub rate_weight_gain_per_hz: f64,
}

impl BrainProbeBinding {
    pub fn new(neuron_id: u64, role: BrainProbeRole, rate_weight_gain_per_hz: f64) -> Result<Self> {
        let binding = Self {
            neuron_id,
            role,
            rate_weight_gain_per_hz,
        };
        binding.validate()?;
        Ok(binding)
    }

    fn validate(self) -> Result<()> {
        if self.neuron_id == 0 {
            bail!("brain probe binding requires an explicit nonzero neuron_id")
        }
        if !self.rate_weight_gain_per_hz.is_finite() {
            bail!("brain probe rate weight must be finite")
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrainProbeRate {
    pub neuron_id: u64,
    pub rate_hz: f64,
}

impl BrainProbeRate {
    pub fn new(neuron_id: u64, rate_hz: f64) -> Result<Self> {
        let probe = Self { neuron_id, rate_hz };
        probe.validate()?;
        Ok(probe)
    }

    fn validate(self) -> Result<()> {
        if self.neuron_id == 0 {
            bail!("brain probe rate requires an explicit nonzero neuron_id")
        }
        if !self.rate_hz.is_finite() || self.rate_hz < 0.0 {
            bail!("brain probe rate_hz must be finite and non-negative")
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BaselineSixLegCpgConfig {
    pub cycle_frequency_hz: f64,
    pub neutral_joint_angles_rad: [f64; JOINTS_PER_LEG],
    pub joint_amplitudes_rad: [f64; JOINTS_PER_LEG],
    pub phase_offsets_rad: [f64; LEG_COUNT],
    pub turn_coxa_offset_rad: f64,
    pub forward_gain_bias: f64,
    pub max_forward_gain: f64,
    pub turn_gain_bias: f64,
    pub max_abs_turn_gain: f64,
    pub probe_bindings: Vec<BrainProbeBinding>,
}

impl Default for BaselineSixLegCpgConfig {
    fn default() -> Self {
        Self {
            cycle_frequency_hz: 2.0,
            neutral_joint_angles_rad: [0.0, 0.45, -0.9],
            joint_amplitudes_rad: [0.25, 0.35, 0.45],
            phase_offsets_rad: [0.0, PI, 0.0, PI, 0.0, PI],
            turn_coxa_offset_rad: 0.15,
            forward_gain_bias: 1.0,
            max_forward_gain: 2.0,
            turn_gain_bias: 0.0,
            max_abs_turn_gain: 1.0,
            probe_bindings: Vec::new(),
        }
    }
}

impl BaselineSixLegCpgConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.cycle_frequency_hz.is_finite() || self.cycle_frequency_hz <= 0.0 {
            bail!("cycle_frequency_hz must be finite and positive")
        }
        if self
            .neutral_joint_angles_rad
            .iter()
            .chain(self.joint_amplitudes_rad.iter())
            .chain(self.phase_offsets_rad.iter())
            .any(|value| !value.is_finite())
        {
            bail!("CPG angle and phase configuration must be finite")
        }
        if self.joint_amplitudes_rad.iter().any(|&value| value < 0.0) {
            bail!("CPG joint amplitudes must be non-negative")
        }
        if !self.turn_coxa_offset_rad.is_finite() {
            bail!("turn_coxa_offset_rad must be finite")
        }
        if !self.forward_gain_bias.is_finite()
            || !self.max_forward_gain.is_finite()
            || self.max_forward_gain <= 0.0
            || self.forward_gain_bias < 0.0
            || self.forward_gain_bias > self.max_forward_gain
        {
            bail!("forward gain bounds are invalid")
        }
        if !self.turn_gain_bias.is_finite()
            || !self.max_abs_turn_gain.is_finite()
            || self.max_abs_turn_gain < 0.0
            || self.turn_gain_bias.abs() > self.max_abs_turn_gain
        {
            bail!("turn gain bounds are invalid")
        }
        let mut binding_keys = HashSet::with_capacity(self.probe_bindings.len());
        for binding in &self.probe_bindings {
            binding.validate()?;
            if !binding_keys.insert((binding.neuron_id, binding.role)) {
                bail!("brain probe bindings must be unique by neuron_id and role")
            }
        }
        Ok(())
    }

    pub fn metadata(&self) -> (&'static str, &'static str, [&'static str; LEG_COUNT]) {
        (
            BASELINE_CPG_MODEL,
            BASELINE_CPG_MAPPING_POLICY,
            BASELINE_LEG_INDEX_ORDER,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SixLegVncCommand {
    pub phase_rad: f64,
    pub forward_gain: f64,
    pub turn_gain: f64,
    pub joint_angles_rad: [[f64; JOINTS_PER_LEG]; LEG_COUNT],
}

/// Engineering six-leg tripod oscillator and VNC command baseline.
///
/// The leg order and phase offsets are exposed through [`Self::metadata`].
/// Each decode evaluates sinusoidal stride/lift waves, applies explicit
/// forward/turn probe bindings, and advances phase by the supplied fixed step.
/// This is a configurable controller placeholder and makes no biological
/// mapping claim.
#[derive(Clone, Debug)]
pub struct BaselineSixLegCpgVncDecoder {
    config: BaselineSixLegCpgConfig,
    phase_rad: f64,
}

impl BaselineSixLegCpgVncDecoder {
    pub fn new(config: BaselineSixLegCpgConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            phase_rad: 0.0,
        })
    }

    pub fn baseline() -> Self {
        Self::new(BaselineSixLegCpgConfig::default()).expect("baseline CPG config is valid")
    }

    pub fn config(&self) -> &BaselineSixLegCpgConfig {
        &self.config
    }

    pub fn metadata(&self) -> (&'static str, &'static str, [&'static str; LEG_COUNT]) {
        self.config.metadata()
    }

    pub fn phase_rad(&self) -> f64 {
        self.phase_rad
    }

    pub fn reset(&mut self) {
        self.phase_rad = 0.0;
    }

    pub fn decode(
        &mut self,
        probe_rates: &[BrainProbeRate],
        dt_ms: f64,
    ) -> Result<SixLegVncCommand> {
        if !dt_ms.is_finite() || dt_ms <= 0.0 {
            bail!("CPG dt_ms must be finite and positive")
        }
        let (forward_gain, turn_gain) = self.probe_gains(probe_rates)?;
        let command_phase = self.phase_rad;
        let mut joint_angles_rad = [[0.0; JOINTS_PER_LEG]; LEG_COUNT];
        for (leg, joints) in joint_angles_rad.iter_mut().enumerate() {
            let leg_phase = command_phase + self.config.phase_offsets_rad[leg];
            let stride_wave = leg_phase.sin();
            let lift_wave = leg_phase.cos();
            let turn_sign = if leg < LEG_COUNT / 2 { 1.0 } else { -1.0 };
            joints[0] = self.config.neutral_joint_angles_rad[0]
                + self.config.joint_amplitudes_rad[0] * forward_gain * stride_wave
                + self.config.turn_coxa_offset_rad * turn_gain * turn_sign;
            joints[1] = self.config.neutral_joint_angles_rad[1]
                + self.config.joint_amplitudes_rad[1] * forward_gain * lift_wave;
            joints[2] = self.config.neutral_joint_angles_rad[2]
                - self.config.joint_amplitudes_rad[2] * forward_gain * stride_wave;
        }
        let phase_increment = 2.0 * PI * self.config.cycle_frequency_hz * dt_ms / 1000.0;
        self.phase_rad = (self.phase_rad + phase_increment).rem_euclid(2.0 * PI);
        Ok(SixLegVncCommand {
            phase_rad: command_phase,
            forward_gain,
            turn_gain,
            joint_angles_rad,
        })
    }

    fn probe_gains(&self, probe_rates: &[BrainProbeRate]) -> Result<(f64, f64)> {
        let mut rates_by_id = BTreeMap::new();
        for probe in probe_rates {
            probe.validate()?;
            if rates_by_id.insert(probe.neuron_id, probe.rate_hz).is_some() {
                bail!("brain probe rates must have unique neuron IDs")
            }
        }
        let mut forward_gain = self.config.forward_gain_bias;
        let mut turn_gain = self.config.turn_gain_bias;
        for binding in &self.config.probe_bindings {
            if let Some(rate_hz) = rates_by_id.get(&binding.neuron_id) {
                let contribution = *rate_hz * binding.rate_weight_gain_per_hz;
                if !contribution.is_finite() {
                    bail!("brain probe contribution must be finite")
                }
                match binding.role {
                    BrainProbeRole::ForwardGain => forward_gain += contribution,
                    BrainProbeRole::TurnGain => turn_gain += contribution,
                }
            }
        }
        Ok((
            forward_gain.clamp(0.0, self.config.max_forward_gain),
            turn_gain.clamp(
                -self.config.max_abs_turn_gain,
                self.config.max_abs_turn_gain,
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_step_scheduler_requires_integer_brain_ratio() {
        let config = FixedStepConfig::new(0.1, 1.0).unwrap();
        assert_eq!(config.brain_steps_per_physics_step, 10);
        let mut scheduler = config.scheduler().unwrap();
        assert_eq!(
            scheduler.next_physics_step().unwrap(),
            StepWindow {
                physics_tick: 0,
                brain_tick_start: 0,
                brain_tick_end_exclusive: 10,
            }
        );
        assert_eq!(scheduler.brain_tick(), 10);
        assert!(FixedStepConfig::new(0.3, 1.0).is_err());
        assert!(FixedStepConfig::new(1.0, 0.1).is_err());
    }

    #[test]
    fn sensory_encoder_is_deterministic_and_sparse() {
        let channel = SensoryRateChannel::new(
            SensoryFeature::FootContact { leg_index: 0 },
            720575940600000001,
            0.0,
            1.0,
            0.0,
            100.0,
        )
        .unwrap();
        let config = SensoryBridgeConfig::new(vec![channel]).unwrap();
        let mut first = SensoryRateEncoder::new(config.clone(), 0.1).unwrap();
        let mut second = SensoryRateEncoder::new(config, 0.1).unwrap();
        let sample = SensorySample {
            foot_contacts: [true, false, false, false, false, false],
            ..SensorySample::default()
        };
        let first_events: Vec<_> = (0..100).map(|_| first.encode(&sample).unwrap()).collect();
        let second_events: Vec<_> = (0..100).map(|_| second.encode(&sample).unwrap()).collect();
        assert_eq!(first_events, second_events);
        assert_eq!(first_events.iter().flatten().count(), 1);
        assert_eq!(
            first_events
                .iter()
                .flatten()
                .next()
                .expect("one event")
                .count,
            1
        );
    }

    #[test]
    fn sensory_sample_rejects_out_of_range_food_odor_activation() {
        let mut sample = SensorySample::default();
        sample.food_odor_activation[0][0] = 1.01;
        assert!(sample.validate().is_err());
        sample.food_odor_activation[0][0] = -0.01;
        assert!(sample.validate().is_err());
    }

    #[test]
    fn food_odor_channel_stays_within_the_orn_rate_range() {
        let channel = SensoryRateChannel::new(
            SensoryFeature::FoodOdor {
                side: AntennaSide::Left,
                band: FoodOdorBand::Attractive,
            },
            720575940600000001,
            0.0,
            1.0,
            8.0,
            192.0,
        )
        .unwrap();
        assert_eq!(channel.rate_hz(&SensorySample::default()), 8.0);
        let mut saturated = SensorySample::default();
        saturated.food_odor_activation[0][0] = 1.0;
        assert_eq!(channel.rate_hz(&saturated), 200.0);
    }

    #[test]
    fn sensory_window_matches_tick_encoding_exactly() {
        let channels = vec![
            SensoryRateChannel::new(
                SensoryFeature::OdorLeft,
                720575940600000001,
                0.0,
                1.0,
                12_000.0,
                8_000.0,
            )
            .unwrap(),
            SensoryRateChannel::new(
                SensoryFeature::OdorLeft,
                720575940600000002,
                0.0,
                1.0,
                10.0,
                90.0,
            )
            .unwrap(),
            SensoryRateChannel::new(
                SensoryFeature::VisualRight,
                720575940600000003,
                0.0,
                1.0,
                5.0,
                75.0,
            )
            .unwrap(),
        ];
        let config = SensoryBridgeConfig::new(channels.clone()).unwrap();
        let mut tick_encoder = SensoryRateEncoder::new(config.clone(), 0.1).unwrap();
        let mut window_encoder = SensoryRateEncoder::new(config, 0.1).unwrap();
        let sample = SensorySample {
            odor_left: 0.75,
            visual_right: 0.4,
            ..SensorySample::default()
        };
        let steps = 137;
        let expected = (0..steps)
            .map(|_| tick_encoder.encode(&sample).unwrap())
            .collect::<Vec<_>>();
        let window = window_encoder.encode_window(&sample, steps).unwrap();
        assert_eq!(window.steps(), steps);
        for (step, expected) in expected.iter().enumerate() {
            let begin = window.step_offsets()[step] as usize;
            let end = window.step_offsets()[step + 1] as usize;
            let actual = window.lanes()[begin..end]
                .iter()
                .zip(&window.counts()[begin..end])
                .map(|(&lane, &count)| SparseExternalEvent {
                    target_neuron_id: channels[lane as usize].target_neuron_id,
                    count,
                })
                .collect::<Vec<_>>();
            assert_eq!(&actual, expected);
        }
        for _ in 0..100 {
            assert_eq!(
                tick_encoder.encode(&sample).unwrap(),
                window_encoder.encode(&sample).unwrap()
            );
        }
    }

    #[test]
    fn angular_speed_channel_responds_to_flight_rotation_magnitude() {
        let channel = SensoryRateChannel::new(
            SensoryFeature::FlightAngularSpeed,
            720575940600000003,
            0.0,
            200.0,
            0.0,
            100.0,
        )
        .unwrap();
        assert_eq!(channel.rate_hz(&SensorySample::default()), 0.0);
        let sample = SensorySample {
            flight_angular_speed_rad_s: 50.0,
            ..SensorySample::default()
        };
        assert_eq!(channel.rate_hz(&sample), 25.0);
    }

    #[test]
    fn population_channels_start_at_deterministic_distinct_phases() {
        let config = SensoryBridgeConfig::new(vec![
            SensoryRateChannel::new(
                SensoryFeature::OdorLeft,
                720575940600000001,
                0.0,
                1.0,
                0.0,
                100.0,
            )
            .unwrap(),
            SensoryRateChannel::new(
                SensoryFeature::OdorLeft,
                720575940600000002,
                0.0,
                1.0,
                0.0,
                100.0,
            )
            .unwrap(),
        ])
        .unwrap();
        let mut encoder = SensoryRateEncoder::new(config, 0.1).unwrap();
        assert_ne!(encoder.phase_cycles[0], encoder.phase_cycles[1]);
        let mut clone = encoder.clone();
        let sample = SensorySample {
            odor_left: 1.0,
            ..SensorySample::default()
        };
        for _ in 0..100 {
            assert_eq!(
                encoder.encode(&sample).unwrap(),
                clone.encode(&sample).unwrap()
            );
        }
    }

    #[test]
    fn sensory_bridge_requires_explicit_target_and_weight() {
        assert!(
            SensoryRateChannel::new(SensoryFeature::OdorIntensity, 0, 0.0, 1.0, 0.0, 1.0,).is_err()
        );
        assert!(
            SensoryRateChannel::new(SensoryFeature::TasteValence, 9, 0.0, 1.0, 0.0, -1.0,).is_err()
        );
        let duplicate =
            SensoryRateChannel::new(SensoryFeature::OdorIntensity, 9, 0.0, 1.0, 0.0, 1.0).unwrap();
        assert!(SensoryBridgeConfig::new(vec![duplicate, duplicate]).is_err());
    }

    #[test]
    fn cpg_probe_rates_drive_explicit_gain_and_turn_inputs() {
        let config = BaselineSixLegCpgConfig {
            probe_bindings: vec![
                BrainProbeBinding::new(101, BrainProbeRole::ForwardGain, 0.01).unwrap(),
                BrainProbeBinding::new(202, BrainProbeRole::TurnGain, 0.02).unwrap(),
            ],
            ..Default::default()
        };
        let mut decoder = BaselineSixLegCpgVncDecoder::new(config).unwrap();
        let command = decoder
            .decode(
                &[
                    BrainProbeRate::new(101, 50.0).unwrap(),
                    BrainProbeRate::new(202, 10.0).unwrap(),
                ],
                0.1,
            )
            .unwrap();
        assert_eq!(command.forward_gain, 1.5);
        assert_eq!(command.turn_gain, 0.2);
        assert_eq!(command.phase_rad, 0.0);
        assert!(command.joint_angles_rad[0][0] != command.joint_angles_rad[3][0]);
        assert_eq!(decoder.metadata().0, BASELINE_CPG_MODEL);
        assert_eq!(decoder.metadata().2, BASELINE_LEG_INDEX_ORDER);
    }

    #[test]
    fn cpg_rejects_implicit_probe_ids_and_duplicate_rates() {
        assert!(BrainProbeBinding::new(0, BrainProbeRole::ForwardGain, 1.0).is_err());
        let binding = BrainProbeBinding::new(11, BrainProbeRole::TurnGain, 1.0).unwrap();
        let config = BaselineSixLegCpgConfig {
            probe_bindings: vec![binding, binding],
            ..Default::default()
        };
        assert!(BaselineSixLegCpgVncDecoder::new(config).is_err());

        let mut decoder = BaselineSixLegCpgVncDecoder::baseline();
        assert!(
            decoder
                .decode(
                    &[
                        BrainProbeRate::new(11, 1.0).unwrap(),
                        BrainProbeRate::new(11, 2.0).unwrap()
                    ],
                    0.1,
                )
                .is_err()
        );
    }
}
