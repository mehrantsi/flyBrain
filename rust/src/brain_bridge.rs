use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::brain_signal::{BrainSignalProcessor, BrainSignalSample, SAMPLE_PERIOD_SECONDS};
#[cfg(target_os = "emscripten")]
use crate::browser_engine::BrowserEngine;
use crate::embodiment::{
    SensoryBridgeConfig, SensoryFeature, SensoryRateChannel, SensoryRateEncoder, SensorySample,
};
#[cfg(target_os = "macos")]
use crate::metal_engine::MetalEngine;
use crate::neural_io::{NeuralIoArtifact, NeuralIoResolution};
use crate::olfaction::{AntennaSide, FoodOdorBand, ORN_MAXIMUM_RATE_HZ, ORN_SPONTANEOUS_RATE_HZ};
use crate::pack::ConnectomePack;
use crate::parameters::ModelParameters;
use crate::protocol::{MN9_LEFT_ID, RIGHT_SUGAR_GRN_IDS};

#[cfg(target_os = "emscripten")]
type NeuralEngine = BrowserEngine;
#[cfg(target_os = "macos")]
type NeuralEngine = MetalEngine;

pub const SUGAR_BRIDGE_MODEL: &str = "flybrain-published-sugar-mn9-v2";
pub const V783_PARTIAL_EMBODIMENT_MODEL: &str =
    "flybrain-v783-connectome-partial-engineered-embodiment-v5";
pub const NEURAL_IO_FILE: &str = "flywire_v783_neural_io.json";
pub const MALE_CNS_NEURAL_IO_FILE: &str = "male_cns_v1_neural_io.json";
pub const MALE_CNS_EMBODIMENT_MODEL: &str =
    "male-cns-v1-connectome-partial-engineered-motor-embodiment-v2";
pub const TASTE_INPUT_RATE_HZ: f64 = 150.0;
pub const OLFACTORY_INPUT_RATE_HZ: f64 = ORN_MAXIMUM_RATE_HZ - ORN_SPONTANEOUS_RATE_HZ;
pub const VISUAL_BASELINE_RATE_HZ: f64 = 5.0;
pub const VISUAL_INPUT_RATE_HZ: f64 = 75.0;
pub const FLIGHT_STATE_INPUT_MAX_RAD_S: f64 = 200.0;
pub const FLIGHT_STATE_INPUT_RATE_HZ: f64 = 120.0;
pub const MOTOR_RATE_FILTER_TAU_MS: f64 = 50.0;
pub const FEEDING_RELEASE_TAU_MS: f64 = 500.0;
pub const FEEDING_EXTENSION_PER_MN9_SPIKE: f64 = 0.35;
pub const POPULATION_TELEMETRY_PERIOD_MS: f64 = 50.0;
pub const DN_DECODER_REFERENCE_RATE_HZ: f64 = 80.0;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct BrainBridgeParameters {
    pub neural: ModelParameters,
    pub taste_input_rate_hz: f64,
    pub olfactory_input_rate_hz: f64,
    #[serde(default = "default_olfactory_baseline_rate_hz")]
    pub olfactory_baseline_rate_hz: f64,
    pub visual_baseline_rate_hz: f64,
    pub visual_input_rate_hz: f64,
    pub flight_state_input_max_rad_s: f64,
    pub flight_state_input_rate_hz: f64,
    pub motor_rate_filter_tau_ms: f64,
    pub feeding_release_tau_ms: f64,
    pub feeding_extension_per_mn9_spike: f64,
    pub dn_decoder_reference_rate_hz: f64,
    pub walking_steering_gain: f64,
    pub flight_steering_gain: f64,
    pub flight_drive_gain: f64,
    #[serde(default = "default_altitude_control_gain")]
    pub altitude_control_gain: f64,
    #[serde(default = "default_cns_motor_outputs_enabled")]
    pub cns_motor_outputs_enabled: bool,
    #[serde(default = "default_cns_motor_outputs_enabled")]
    pub cns_landing_output_enabled: bool,
    #[serde(default = "default_cns_motor_reference_rate_hz")]
    pub cns_motor_reference_rate_hz: f64,
}

fn default_cns_motor_outputs_enabled() -> bool {
    true
}
fn default_cns_motor_reference_rate_hz() -> f64 {
    20.0
}

fn default_altitude_control_gain() -> f64 {
    1.0
}

fn default_olfactory_baseline_rate_hz() -> f64 {
    ORN_SPONTANEOUS_RATE_HZ
}

impl Default for BrainBridgeParameters {
    fn default() -> Self {
        Self {
            neural: ModelParameters::default(),
            taste_input_rate_hz: TASTE_INPUT_RATE_HZ,
            olfactory_input_rate_hz: OLFACTORY_INPUT_RATE_HZ,
            olfactory_baseline_rate_hz: default_olfactory_baseline_rate_hz(),
            visual_baseline_rate_hz: VISUAL_BASELINE_RATE_HZ,
            visual_input_rate_hz: VISUAL_INPUT_RATE_HZ,
            flight_state_input_max_rad_s: FLIGHT_STATE_INPUT_MAX_RAD_S,
            flight_state_input_rate_hz: FLIGHT_STATE_INPUT_RATE_HZ,
            motor_rate_filter_tau_ms: MOTOR_RATE_FILTER_TAU_MS,
            feeding_release_tau_ms: FEEDING_RELEASE_TAU_MS,
            feeding_extension_per_mn9_spike: FEEDING_EXTENSION_PER_MN9_SPIKE,
            dn_decoder_reference_rate_hz: DN_DECODER_REFERENCE_RATE_HZ,
            walking_steering_gain: 1.0,
            flight_steering_gain: 1.0,
            flight_drive_gain: 1.0,
            altitude_control_gain: 1.0,
            cns_motor_outputs_enabled: true,
            cns_landing_output_enabled: true,
            cns_motor_reference_rate_hz: default_cns_motor_reference_rate_hz(),
        }
    }
}

impl BrainBridgeParameters {
    pub fn validate(self) -> Result<Self> {
        self.neural.validate()?;
        if [
            self.taste_input_rate_hz,
            self.olfactory_input_rate_hz,
            self.olfactory_baseline_rate_hz,
            self.visual_baseline_rate_hz,
            self.visual_input_rate_hz,
            self.flight_state_input_rate_hz,
            self.feeding_extension_per_mn9_spike,
            self.walking_steering_gain,
            self.flight_steering_gain,
            self.flight_drive_gain,
            self.altitude_control_gain,
        ]
        .into_iter()
        .any(|value| !value.is_finite() || value < 0.0)
            || [
                self.flight_state_input_max_rad_s,
                self.motor_rate_filter_tau_ms,
                self.feeding_release_tau_ms,
                self.dn_decoder_reference_rate_hz,
                self.cns_motor_reference_rate_hz,
            ]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
        {
            bail!("brain bridge parameters are invalid")
        }
        if self.olfactory_baseline_rate_hz > ORN_MAXIMUM_RATE_HZ {
            bail!("olfactory baseline rate exceeds the maximum ORN rate")
        }
        if self.olfactory_baseline_rate_hz + self.olfactory_input_rate_hz > ORN_MAXIMUM_RATE_HZ {
            bail!("olfactory baseline and evoked rates exceed the maximum ORN rate")
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SensoryClass {
    Taste,
    Olfactory,
    Visual,
    FlightState,
}

pub struct BrainBodyBridge {
    parameters: BrainBridgeParameters,
    engine: NeuralEngine,
    encoder: SensoryRateEncoder,
    sensory_ids: Vec<u64>,
    sensory_indices: Vec<u32>,
    sensory_class_by_lane: Vec<SensoryClass>,
    probe_indices: Vec<u32>,
    walking_left_probe_positions: Vec<usize>,
    walking_right_probe_positions: Vec<usize>,
    flight_left_probe_positions: Vec<usize>,
    flight_right_probe_positions: Vec<usize>,
    flight_power_increase_probe_positions: Vec<usize>,
    flight_power_decrease_probe_positions: Vec<usize>,
    landing_probe_positions: Vec<usize>,
    motor_neuron_id: u64,
    filtered_mn9_rate_hz: f64,
    filtered_population_rate_hz: f64,
    filtered_walking_left_rate_hz: f64,
    filtered_walking_right_rate_hz: f64,
    filtered_flight_left_rate_hz: f64,
    filtered_flight_right_rate_hz: f64,
    filtered_flight_power_increase_rate_hz: f64,
    filtered_flight_power_decrease_rate_hz: f64,
    filtered_landing_rate_hz: f64,
    previous_total_spikes: u64,
    spiking_neuron_count: usize,
    population_telemetry_elapsed_ms: f64,
    brain_signal: BrainSignalProcessor,
    brain_signal_elapsed_ms: f64,
    brain_signal_sequence: u64,
    latest_brain_signal: BrainSignalSample,
    telemetry_enabled: bool,
    feeding_extension: f64,
    neuron_count: usize,
    brain_dt_ms: f64,
    full_neural_io: bool,
    neural_io_stats: NeuralIoStats,
    cns_motor_probes: Option<CnsMotorProbes>,
    cns_olfactory_probes: Option<crate::cns_olfaction::CnsOlfactoryProbes>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct CnsMotorReadout {
    pub spike_delta: u64,
    pub flight_power_hz: [f64; 2],
    pub wing_steering_hz: [f64; 2],
    pub walking_hz: [f64; 2],
    pub landing_hz: [f64; 2],
    pub flight_activation: f64,
    pub walking_activation: f64,
    pub steering: f64,
    pub outputs_connected: bool,
}

struct CnsMotorProbes {
    positions: [[Vec<usize>; 2]; 4],
    unique_positions: Vec<usize>,
    filtered_rates: [[f64; 2]; 4],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NeuralIoStats {
    pub groups: usize,
    pub selected_root_ids: usize,
    pub present_root_ids: usize,
    pub missing_root_ids: usize,
}

#[derive(Clone, Debug)]
pub struct BrainWindowResult {
    pub elapsed: Duration,
    pub encoding_elapsed: Duration,
    pub engine_elapsed: Duration,
    pub external_event_count: u64,
    pub taste_event_count: u64,
    pub olfactory_event_count: u64,
    pub visual_event_count: u64,
    pub flight_state_event_count: u64,
    pub mn9_spike_delta: u32,
    pub mn9_rate_hz: f64,
    pub filtered_mn9_rate_hz: f64,
    pub population_spike_delta: u64,
    pub cumulative_spiking_neuron_count: usize,
    pub filtered_population_rate_hz: f64,
    pub brain_field_potential_mv: f64,
    pub brain_field_dominant_frequency_hz: f64,
    pub brain_field_sample_sequence: u64,
    pub walking_left_rate_hz: f64,
    pub walking_right_rate_hz: f64,
    pub flight_left_rate_hz: f64,
    pub flight_right_rate_hz: f64,
    pub flight_power_increase_rate_hz: f64,
    pub flight_power_decrease_rate_hz: f64,
    pub landing_dn_rate_hz: f64,
    pub brain_walking_drive: f64,
    pub brain_walking_steering: f64,
    pub brain_flight_drive: f64,
    pub brain_flight_steering: f64,
    pub brain_altitude_control: f64,
    pub brain_landing_drive: f64,
    pub feeding_extension: f64,
    pub forward_gain: f64,
    pub turn_gain: f64,
    pub cns_motor: Option<CnsMotorReadout>,
    pub cns_olfactory: Option<crate::cns_olfaction::CnsOlfactoryReadout>,
}

impl BrainBodyBridge {
    pub fn new(pack: &ConnectomePack) -> Result<Self> {
        Self::build(pack, None, BrainBridgeParameters::default())
    }

    pub fn new_with_parameters(
        pack: &ConnectomePack,
        parameters: BrainBridgeParameters,
    ) -> Result<Self> {
        Self::build(pack, None, parameters)
    }

    pub fn new_with_neural_io(
        pack: &ConnectomePack,
        artifact_path: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::new_with_neural_io_and_parameters(
            pack,
            artifact_path,
            BrainBridgeParameters::default(),
        )
    }

    pub fn new_with_neural_io_and_parameters(
        pack: &ConnectomePack,
        artifact_path: impl AsRef<Path>,
        parameters: BrainBridgeParameters,
    ) -> Result<Self> {
        let artifact = NeuralIoArtifact::load(artifact_path)?;
        Self::build(pack, Some(artifact.resolve(pack)?), parameters)
    }

    fn build(
        pack: &ConnectomePack,
        neural_io: Option<NeuralIoResolution>,
        parameters: BrainBridgeParameters,
    ) -> Result<Self> {
        let parameters = parameters.validate()?;
        let neural_io_stats = neural_io.as_ref().map(resolution_stats).unwrap_or_default();
        let male_cns = neural_io
            .as_ref()
            .is_some_and(|io| io.artifact.is_male_cns());
        let mut channels = Vec::new();
        let mut sensory_ids = Vec::new();
        let mut sensory_indices = Vec::new();
        let mut sensory_classes = Vec::new();

        let taste_ids: &[u64] = if male_cns {
            &neural_io
                .as_ref()
                .unwrap()
                .group("taste_sugar")
                .context("MaleCNS neural I/O is missing taste_sugar")?
                .selected_root_ids
        } else {
            &RIGHT_SUGAR_GRN_IDS
        };
        for &neuron_id in taste_ids {
            if let Some(index) = neuron_index(pack, neuron_id) {
                channels.push(SensoryRateChannel::new(
                    SensoryFeature::TasteValence,
                    neuron_id,
                    0.0,
                    1.0,
                    0.0,
                    parameters.taste_input_rate_hz,
                )?);
                sensory_ids.push(neuron_id);
                sensory_indices.push(index);
                sensory_classes.push(SensoryClass::Taste);
            }
        }
        if sensory_ids.is_empty() {
            bail!("connectome contains none of the configured sugar GRNs")
        }

        if let Some(resolution) = neural_io.as_ref() {
            let food_olfaction = food_olfaction_bindings(resolution)?;
            append_olfactory_channels(
                pack,
                resolution,
                "olfaction_left",
                AntennaSide::Left,
                &food_olfaction,
                parameters.olfactory_baseline_rate_hz,
                parameters.olfactory_input_rate_hz,
                SensoryClass::Olfactory,
                &mut channels,
                &mut sensory_ids,
                &mut sensory_indices,
                &mut sensory_classes,
            )?;
            append_olfactory_channels(
                pack,
                resolution,
                "olfaction_right",
                AntennaSide::Right,
                &food_olfaction,
                parameters.olfactory_baseline_rate_hz,
                parameters.olfactory_input_rate_hz,
                SensoryClass::Olfactory,
                &mut channels,
                &mut sensory_ids,
                &mut sensory_indices,
                &mut sensory_classes,
            )?;
            for group in ["visual_left_r1_6", "visual_left_r7", "visual_left_r8"] {
                if male_cns {
                    continue;
                }
                append_population_channels(
                    pack,
                    resolution,
                    group,
                    SensoryFeature::VisualLeft,
                    parameters.visual_baseline_rate_hz,
                    parameters.visual_input_rate_hz,
                    SensoryClass::Visual,
                    &mut channels,
                    &mut sensory_ids,
                    &mut sensory_indices,
                    &mut sensory_classes,
                )?;
            }
            for group in ["visual_right_r1_6", "visual_right_r7", "visual_right_r8"] {
                if male_cns {
                    continue;
                }
                append_population_channels(
                    pack,
                    resolution,
                    group,
                    SensoryFeature::VisualRight,
                    parameters.visual_baseline_rate_hz,
                    parameters.visual_input_rate_hz,
                    SensoryClass::Visual,
                    &mut channels,
                    &mut sensory_ids,
                    &mut sensory_indices,
                    &mut sensory_classes,
                )?;
            }
            for group in [
                "flight_state_msahn_left",
                "flight_state_msahn_right",
                "flight_state_mtahn_left",
                "flight_state_mtahn_right",
                "flight_state_sapp_left",
                "flight_state_sapp_right",
            ] {
                if resolution.group(group).is_none() && (male_cns || group.contains("sapp")) {
                    continue;
                }
                append_flight_state_channels(
                    pack,
                    resolution,
                    group,
                    parameters.flight_state_input_max_rad_s,
                    parameters.flight_state_input_rate_hz,
                    &mut channels,
                    &mut sensory_ids,
                    &mut sensory_indices,
                    &mut sensory_classes,
                )?;
            }
            for group in ["flight_jo_e_left", "flight_jo_e_right"] {
                if male_cns && resolution.group(group).is_none() {
                    continue;
                }
                append_population_channels(
                    pack,
                    resolution,
                    group,
                    SensoryFeature::FlightMechanosensory,
                    0.0,
                    parameters.flight_state_input_rate_hz,
                    SensoryClass::FlightState,
                    &mut channels,
                    &mut sensory_ids,
                    &mut sensory_indices,
                    &mut sensory_classes,
                )?;
            }
            if male_cns {
                for (group, feature) in [
                    ("visual_motion_left", SensoryFeature::VisualMotionLeft),
                    ("visual_motion_right", SensoryFeature::VisualMotionRight),
                    ("visual_loom_left", SensoryFeature::VisualLoomLeft),
                    ("visual_loom_right", SensoryFeature::VisualLoomRight),
                ] {
                    append_population_channels(
                        pack,
                        resolution,
                        group,
                        feature,
                        parameters.visual_baseline_rate_hz,
                        parameters.visual_input_rate_hz,
                        SensoryClass::Visual,
                        &mut channels,
                        &mut sensory_ids,
                        &mut sensory_indices,
                        &mut sensory_classes,
                    )?;
                }
            }
        }

        let encoder =
            SensoryRateEncoder::new(SensoryBridgeConfig::new(channels)?, parameters.neural.dt_ms)?;
        let motor_neuron_id = if male_cns {
            let group = neural_io
                .as_ref()
                .unwrap()
                .group("feeding_mn9")
                .context("MaleCNS neural I/O is missing feeding_mn9")?;
            if group.selected_root_ids.len() != 1 || !group.missing_root_ids.is_empty() {
                bail!("MaleCNS feeding_mn9 must identify one present feeding motor neuron")
            }
            group.selected_root_ids[0]
        } else {
            MN9_LEFT_ID
        };
        let motor_neuron_index = neuron_index(pack, motor_neuron_id).with_context(|| {
            format!("connectome is missing contralateral MN9 {motor_neuron_id}")
        })?;
        let mut probe_indices = vec![motor_neuron_index];
        let mut probe_position_by_index = BTreeMap::from([(motor_neuron_index, 0_usize)]);
        let (walking_left_probe_positions, walking_right_probe_positions) =
            collect_bilateral_probe_positions(
                neural_io.as_ref(),
                "walking_",
                &[],
                &mut probe_indices,
                &mut probe_position_by_index,
            );
        let (flight_left_probe_positions, flight_right_probe_positions) =
            collect_bilateral_probe_positions(
                neural_io.as_ref(),
                "flight_",
                &[
                    "flight_state_",
                    "flight_jo_e_",
                    "flight_dng02_",
                    "flight_dng07_",
                ],
                &mut probe_indices,
                &mut probe_position_by_index,
            );
        let flight_power_increase_probe_positions = collect_probe_positions(
            neural_io.as_ref(),
            "flight_dng02_",
            &mut probe_indices,
            &mut probe_position_by_index,
        );
        let flight_power_decrease_probe_positions = collect_probe_positions(
            neural_io.as_ref(),
            "flight_dng07_",
            &mut probe_indices,
            &mut probe_position_by_index,
        );
        let landing_probe_positions = collect_probe_positions(
            neural_io.as_ref(),
            "landing_",
            &mut probe_indices,
            &mut probe_position_by_index,
        );
        let cns_motor_probes = if male_cns {
            Some(CnsMotorProbes::new(
                neural_io.as_ref().unwrap(),
                &mut probe_indices,
                &mut probe_position_by_index,
            )?)
        } else {
            None
        };
        let engine = NeuralEngine::new(pack, parameters.neural, None, None, &sensory_indices, &[])?;
        let cns_olfactory_probes = if male_cns {
            Some(crate::cns_olfaction::CnsOlfactoryProbes::new(
                pack,
                neural_io.as_ref().unwrap(),
                &mut probe_indices,
                &mut probe_position_by_index,
            )?)
        } else {
            None
        };
        Ok(Self {
            parameters,
            engine,
            encoder,
            sensory_ids,
            sensory_indices,
            sensory_class_by_lane: sensory_classes,
            probe_indices,
            walking_left_probe_positions,
            walking_right_probe_positions,
            flight_left_probe_positions,
            flight_right_probe_positions,
            flight_power_increase_probe_positions,
            flight_power_decrease_probe_positions,
            landing_probe_positions,
            motor_neuron_id,
            filtered_mn9_rate_hz: 0.0,
            filtered_population_rate_hz: 0.0,
            filtered_walking_left_rate_hz: 0.0,
            filtered_walking_right_rate_hz: 0.0,
            filtered_flight_left_rate_hz: 0.0,
            filtered_flight_right_rate_hz: 0.0,
            filtered_flight_power_increase_rate_hz: 0.0,
            filtered_flight_power_decrease_rate_hz: 0.0,
            filtered_landing_rate_hz: 0.0,
            previous_total_spikes: 0,
            spiking_neuron_count: 0,
            population_telemetry_elapsed_ms: 0.0,
            brain_signal: BrainSignalProcessor::new(),
            brain_signal_elapsed_ms: 0.0,
            brain_signal_sequence: 0,
            latest_brain_signal: BrainSignalSample::default(),
            telemetry_enabled: false,
            feeding_extension: 0.0,
            neuron_count: pack.neuron_count(),
            brain_dt_ms: parameters.neural.dt_ms,
            full_neural_io: neural_io.is_some(),
            neural_io_stats,
            cns_motor_probes,
            cns_olfactory_probes,
        })
    }

    pub fn run_window(
        &mut self,
        sample: &SensorySample,
        brain_steps: usize,
    ) -> Result<BrainWindowResult> {
        if brain_steps == 0 {
            bail!("brain window must contain at least one step")
        }
        let started = Instant::now();
        let sensory_window = self.encoder.encode_window(sample, brain_steps)?;
        let mut class_event_counts = [0_u64; 4];
        for (&lane, &count) in sensory_window.lanes().iter().zip(sensory_window.counts()) {
            class_event_counts[class_index(self.sensory_class_by_lane[lane as usize])] +=
                u64::from(count);
        }
        let encoding_elapsed = started.elapsed();
        let window = self.engine.run_window_sparse(
            sensory_window.steps(),
            sensory_window.step_offsets(),
            sensory_window.lanes(),
            sensory_window.counts(),
            &self.probe_indices,
        )?;
        let mn9_spike_delta = *window
            .spike_count_deltas
            .first()
            .context("MN9 probe result is missing")?;
        let window_ms = brain_steps as f64 * self.brain_dt_ms;
        let mn9_rate_hz = f64::from(mn9_spike_delta) * 1000.0 / window_ms;
        let alpha = 1.0 - (-window_ms / self.parameters.motor_rate_filter_tau_ms).exp();
        self.filtered_mn9_rate_hz += alpha * (mn9_rate_hz - self.filtered_mn9_rate_hz);

        let walking_left_rate_hz = probe_population_rate(
            &window.spike_count_deltas,
            &self.walking_left_probe_positions,
            window_ms,
        );
        let walking_right_rate_hz = probe_population_rate(
            &window.spike_count_deltas,
            &self.walking_right_probe_positions,
            window_ms,
        );
        let flight_left_rate_hz = probe_population_rate(
            &window.spike_count_deltas,
            &self.flight_left_probe_positions,
            window_ms,
        );
        let flight_right_rate_hz = probe_population_rate(
            &window.spike_count_deltas,
            &self.flight_right_probe_positions,
            window_ms,
        );
        let flight_power_increase_rate_hz = probe_population_rate(
            &window.spike_count_deltas,
            &self.flight_power_increase_probe_positions,
            window_ms,
        );
        let flight_power_decrease_rate_hz = probe_population_rate(
            &window.spike_count_deltas,
            &self.flight_power_decrease_probe_positions,
            window_ms,
        );
        let landing_dn_rate_hz = probe_population_rate(
            &window.spike_count_deltas,
            &self.landing_probe_positions,
            window_ms,
        );
        update_filtered_rate(
            &mut self.filtered_walking_left_rate_hz,
            walking_left_rate_hz,
            alpha,
        );
        update_filtered_rate(
            &mut self.filtered_walking_right_rate_hz,
            walking_right_rate_hz,
            alpha,
        );
        update_filtered_rate(
            &mut self.filtered_flight_left_rate_hz,
            flight_left_rate_hz,
            alpha,
        );
        update_filtered_rate(
            &mut self.filtered_flight_right_rate_hz,
            flight_right_rate_hz,
            alpha,
        );
        update_filtered_rate(
            &mut self.filtered_flight_power_increase_rate_hz,
            flight_power_increase_rate_hz,
            alpha,
        );
        update_filtered_rate(
            &mut self.filtered_flight_power_decrease_rate_hz,
            flight_power_decrease_rate_hz,
            alpha,
        );
        update_filtered_rate(
            &mut self.filtered_landing_rate_hz,
            landing_dn_rate_hz,
            alpha,
        );

        self.population_telemetry_elapsed_ms += window_ms;
        self.brain_signal_elapsed_ms += window_ms;
        if self.telemetry_enabled
            && self.brain_signal_elapsed_ms + f64::EPSILON >= SAMPLE_PERIOD_SECONDS * 1000.0
        {
            let mean_voltage_deviation_mv = self
                .engine
                .mean_voltage_deviation_mv(self.parameters.neural.resting_mv);
            self.latest_brain_signal = self.brain_signal.update(mean_voltage_deviation_mv);
            self.brain_signal_sequence = self.brain_signal_sequence.wrapping_add(1);
            self.brain_signal_elapsed_ms %= SAMPLE_PERIOD_SECONDS * 1000.0;
        }
        let (population_spike_delta, filtered_population_rate_hz) = if self.telemetry_enabled
            && self.population_telemetry_elapsed_ms >= POPULATION_TELEMETRY_PERIOD_MS
        {
            let total_spikes = self.engine.total_spike_count();
            let delta = total_spikes
                .checked_sub(self.previous_total_spikes)
                .context("whole-brain spike count moved backwards")?;
            self.previous_total_spikes = total_spikes;
            self.spiking_neuron_count = self.engine.spiking_neuron_count();
            let rate_hz = delta as f64 * 1000.0 / self.population_telemetry_elapsed_ms;
            let population_alpha = 1.0
                - (-self.population_telemetry_elapsed_ms
                    / self.parameters.motor_rate_filter_tau_ms)
                    .exp();
            self.filtered_population_rate_hz +=
                population_alpha * (rate_hz - self.filtered_population_rate_hz);
            self.population_telemetry_elapsed_ms = 0.0;
            (delta, self.filtered_population_rate_hz)
        } else {
            (0, self.filtered_population_rate_hz)
        };

        self.feeding_extension = next_feeding_extension(
            self.feeding_extension,
            contextual_feeding_spikes(self.is_male_cns(), sample.taste_valence, mn9_spike_delta),
            window_ms,
            self.parameters.feeding_release_tau_ms,
            self.parameters.feeding_extension_per_mn9_spike,
        );
        let mut brain_walking_steering = bilateral_decoder(
            self.filtered_walking_left_rate_hz,
            self.filtered_walking_right_rate_hz,
            self.parameters.dn_decoder_reference_rate_hz,
            self.parameters.walking_steering_gain,
        );
        let mut brain_walking_drive = normalized_population_drive(
            0.5 * (self.filtered_walking_left_rate_hz + self.filtered_walking_right_rate_hz),
            self.parameters.dn_decoder_reference_rate_hz,
        );
        let mut brain_flight_steering = bilateral_decoder(
            self.filtered_flight_left_rate_hz,
            self.filtered_flight_right_rate_hz,
            self.parameters.dn_decoder_reference_rate_hz,
            self.parameters.flight_steering_gain,
        );
        let mut brain_flight_drive = (self.parameters.flight_drive_gain
            * (self.filtered_flight_left_rate_hz + self.filtered_flight_right_rate_hz)
            / (2.0 * self.parameters.dn_decoder_reference_rate_hz))
            .clamp(0.0, 1.0);
        let mut brain_altitude_control = antagonistic_decoder(
            self.filtered_flight_power_increase_rate_hz,
            self.filtered_flight_power_decrease_rate_hz,
            self.parameters.dn_decoder_reference_rate_hz,
            self.parameters.altitude_control_gain,
        );
        let mut brain_landing_drive = normalized_population_drive(
            self.filtered_landing_rate_hz,
            self.parameters.dn_decoder_reference_rate_hz,
        );
        let cns_motor = self.cns_motor_probes.as_mut().map(|probes| {
            probes.update(
                &window.spike_count_deltas,
                window_ms,
                alpha,
                self.parameters,
            )
        });
        let cns_olfactory = self.cns_olfactory_probes.as_mut().map(|probes| {
            probes.update(
                &window.spike_count_deltas,
                window_ms,
                self.parameters.olfactory_baseline_rate_hz,
            )
        });
        if let Some(motor) = cns_motor {
            brain_walking_drive = motor.walking_activation;
            brain_walking_steering *= motor.walking_activation;
            brain_flight_drive = motor.flight_activation;
            brain_flight_steering = motor.steering;
            brain_altitude_control *= motor.flight_activation;
            brain_landing_drive *= normalized_population_drive(
                0.5 * (motor.landing_hz[0] + motor.landing_hz[1]),
                self.parameters.cns_motor_reference_rate_hz,
            );
            if !self.parameters.cns_landing_output_enabled {
                brain_landing_drive = 0.0;
            }
            if !motor.outputs_connected {
                brain_landing_drive = 0.0;
                self.feeding_extension = 0.0;
            }
        }
        Ok(BrainWindowResult {
            elapsed: started.elapsed(),
            encoding_elapsed,
            engine_elapsed: window.elapsed,
            external_event_count: class_event_counts.iter().sum(),
            taste_event_count: class_event_counts[0],
            olfactory_event_count: class_event_counts[1],
            visual_event_count: class_event_counts[2],
            flight_state_event_count: class_event_counts[3],
            mn9_spike_delta,
            mn9_rate_hz,
            filtered_mn9_rate_hz: self.filtered_mn9_rate_hz,
            population_spike_delta,
            cumulative_spiking_neuron_count: self.spiking_neuron_count,
            filtered_population_rate_hz,
            brain_field_potential_mv: self.latest_brain_signal.filtered_field_mv,
            brain_field_dominant_frequency_hz: self.latest_brain_signal.dominant_frequency_hz,
            brain_field_sample_sequence: self.brain_signal_sequence,
            walking_left_rate_hz: self.filtered_walking_left_rate_hz,
            walking_right_rate_hz: self.filtered_walking_right_rate_hz,
            flight_left_rate_hz: self.filtered_flight_left_rate_hz,
            flight_right_rate_hz: self.filtered_flight_right_rate_hz,
            flight_power_increase_rate_hz: self.filtered_flight_power_increase_rate_hz,
            flight_power_decrease_rate_hz: self.filtered_flight_power_decrease_rate_hz,
            landing_dn_rate_hz: self.filtered_landing_rate_hz,
            brain_walking_drive,
            brain_walking_steering,
            brain_flight_drive,
            brain_flight_steering,
            brain_altitude_control,
            brain_landing_drive,
            feeding_extension: self.feeding_extension,
            forward_gain: brain_walking_drive * (1.0 - self.feeding_extension),
            turn_gain: brain_walking_steering,
            cns_motor,
            cns_olfactory,
        })
    }

    pub fn model_name(&self) -> &'static str {
        if self.cns_motor_probes.is_some() {
            MALE_CNS_EMBODIMENT_MODEL
        } else if self.full_neural_io {
            V783_PARTIAL_EMBODIMENT_MODEL
        } else {
            SUGAR_BRIDGE_MODEL
        }
    }

    pub fn is_male_cns(&self) -> bool {
        self.cns_motor_probes.is_some()
    }

    pub fn device_name(&self) -> &str {
        self.engine.device_name()
    }

    pub fn allocated_bytes(&self) -> usize {
        self.engine.allocated_bytes()
    }

    pub fn sensory_neuron_ids(&self) -> &[u64] {
        &self.sensory_ids
    }

    pub fn motor_neuron_id(&self) -> u64 {
        self.motor_neuron_id
    }

    pub fn full_neural_io_enabled(&self) -> bool {
        self.full_neural_io
    }

    pub fn neuron_count(&self) -> usize {
        self.neuron_count
    }

    pub fn sensory_neuron_count(&self) -> usize {
        self.sensory_indices.len()
    }

    pub fn neural_io_stats(&self) -> NeuralIoStats {
        self.neural_io_stats
    }

    pub fn parameters(&self) -> BrainBridgeParameters {
        self.parameters
    }

    pub fn set_telemetry_enabled(&mut self, enabled: bool) {
        if enabled && !self.telemetry_enabled {
            self.previous_total_spikes = self.engine.total_spike_count();
            self.spiking_neuron_count = self.engine.spiking_neuron_count();
            self.population_telemetry_elapsed_ms = 0.0;
            self.filtered_population_rate_hz = 0.0;
            self.brain_signal.reset();
            self.brain_signal_elapsed_ms = 0.0;
            self.brain_signal_sequence = 0;
            self.latest_brain_signal = BrainSignalSample::default();
        }
        self.telemetry_enabled = enabled;
    }
}

fn resolution_stats(resolution: &NeuralIoResolution) -> NeuralIoStats {
    NeuralIoStats {
        groups: resolution.groups.len(),
        selected_root_ids: resolution
            .groups
            .values()
            .map(|group| group.selected_root_ids.len())
            .sum(),
        present_root_ids: resolution
            .groups
            .values()
            .map(|group| group.engine_indices.len())
            .sum(),
        missing_root_ids: resolution
            .groups
            .values()
            .map(|group| group.missing_root_ids.len())
            .sum(),
    }
}

impl CnsMotorProbes {
    fn new(
        resolution: &NeuralIoResolution,
        probe_indices: &mut Vec<u32>,
        position_by_index: &mut BTreeMap<u32, usize>,
    ) -> Result<Self> {
        let mut positions: [[Vec<usize>; 2]; 4] = Default::default();
        for (pool, prefix) in [
            "motor_flight_power_",
            "motor_flight_steering_",
            "motor_walking_",
            "motor_landing_",
        ]
        .into_iter()
        .enumerate()
        {
            for side in ["left", "right"] {
                let name = format!("{prefix}{side}");
                let group = resolution
                    .group(&name)
                    .with_context(|| format!("MaleCNS is missing motor pool {name}"))?;
                if group.engine_indices.is_empty() || !group.missing_root_ids.is_empty() {
                    bail!("MaleCNS motor pool {name} must resolve completely")
                }
            }
            let (left, right) = collect_bilateral_probe_positions(
                Some(resolution),
                prefix,
                &[],
                probe_indices,
                position_by_index,
            );
            positions[pool] = [left, right];
        }
        let unique_positions = positions
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        Ok(Self {
            positions,
            unique_positions,
            filtered_rates: [[0.0; 2]; 4],
        })
    }

    fn update(
        &mut self,
        deltas: &[u32],
        window_ms: f64,
        alpha: f64,
        parameters: BrainBridgeParameters,
    ) -> CnsMotorReadout {
        for (positions, filtered) in self.positions.iter().zip(&mut self.filtered_rates) {
            for side in 0..2 {
                update_filtered_rate(
                    &mut filtered[side],
                    probe_population_rate(deltas, &positions[side], window_ms),
                    alpha,
                );
            }
        }
        let [flight_power_hz, wing_steering_hz, walking_hz, landing_hz] = self.filtered_rates;
        let connected = if parameters.cns_motor_outputs_enabled {
            1.0
        } else {
            0.0
        };
        CnsMotorReadout {
            spike_delta: self
                .unique_positions
                .iter()
                .map(|&position| u64::from(deltas[position]))
                .sum(),
            flight_power_hz,
            wing_steering_hz,
            walking_hz,
            landing_hz,
            flight_activation: connected
                * cns_motor_activation(
                    0.5 * (flight_power_hz[0] + flight_power_hz[1]),
                    parameters.cns_motor_reference_rate_hz,
                ),
            walking_activation: connected
                * cns_motor_activation(
                    0.5 * (walking_hz[0] + walking_hz[1]),
                    parameters.cns_motor_reference_rate_hz,
                ),
            steering: connected
                * bilateral_decoder(
                    wing_steering_hz[0],
                    wing_steering_hz[1],
                    parameters.cns_motor_reference_rate_hz,
                    parameters.flight_steering_gain,
                ),
            outputs_connected: parameters.cns_motor_outputs_enabled,
        }
    }
}

fn food_olfaction_bindings(
    resolution: &NeuralIoResolution,
) -> Result<BTreeMap<u64, (AntennaSide, FoodOdorBand)>> {
    let profiles = resolution
        .artifact
        .food_olfaction
        .as_ref()
        .context("full neural I/O artifact is missing food-olfaction profiles")?;
    let mut bindings = BTreeMap::new();
    for channel in profiles.channels.values() {
        let side = match channel.side.as_str() {
            "left" => AntennaSide::Left,
            "right" => AntennaSide::Right,
            other => bail!("unsupported food-olfaction channel side {other:?}"),
        };
        let band = FoodOdorBand::from_profile_name(&channel.response_band)?;
        for &root_id in &channel.root_ids {
            if bindings.insert(root_id, (side, band)).is_some() {
                bail!("food-olfaction root {root_id} appears in more than one channel")
            }
        }
    }
    Ok(bindings)
}

#[allow(clippy::too_many_arguments)]
fn append_olfactory_channels(
    pack: &ConnectomePack,
    resolution: &NeuralIoResolution,
    group_name: &str,
    side: AntennaSide,
    food_olfaction: &BTreeMap<u64, (AntennaSide, FoodOdorBand)>,
    baseline_rate_hz: f64,
    evoked_rate_hz: f64,
    class: SensoryClass,
    channels: &mut Vec<SensoryRateChannel>,
    sensory_ids: &mut Vec<u64>,
    sensory_indices: &mut Vec<u32>,
    sensory_classes: &mut Vec<SensoryClass>,
) -> Result<()> {
    let group = resolution
        .group(group_name)
        .with_context(|| format!("neural I/O artifact is missing group {group_name}"))?;
    if baseline_rate_hz < 0.0
        || evoked_rate_hz < 0.0
        || baseline_rate_hz + evoked_rate_hz > ORN_MAXIMUM_RATE_HZ
    {
        bail!("olfactory baseline and evoked rates exceed the maximum ORN rate")
    }
    for &index in &group.engine_indices {
        let neuron_id = pack.neuron_ids[index as usize];
        let (feature, rate_weight_hz_per_unit) = match food_olfaction.get(&neuron_id) {
            Some(&(mapped_side, band)) => {
                if mapped_side != side {
                    bail!("food-olfaction root {neuron_id} is mapped to the wrong antenna side")
                }
                (SensoryFeature::FoodOdor { side, band }, evoked_rate_hz)
            }
            None => (
                match side {
                    AntennaSide::Left => SensoryFeature::OdorLeft,
                    AntennaSide::Right => SensoryFeature::OdorRight,
                },
                0.0,
            ),
        };
        channels.push(SensoryRateChannel::new(
            feature,
            neuron_id,
            0.0,
            1.0,
            baseline_rate_hz,
            rate_weight_hz_per_unit,
        )?);
        sensory_ids.push(neuron_id);
        sensory_indices.push(index);
        sensory_classes.push(class);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_population_channels(
    pack: &ConnectomePack,
    resolution: &NeuralIoResolution,
    group_name: &str,
    feature: SensoryFeature,
    baseline_rate_hz: f64,
    rate_weight_hz_per_unit: f64,
    class: SensoryClass,
    channels: &mut Vec<SensoryRateChannel>,
    sensory_ids: &mut Vec<u64>,
    sensory_indices: &mut Vec<u32>,
    sensory_classes: &mut Vec<SensoryClass>,
) -> Result<()> {
    let group = resolution
        .group(group_name)
        .with_context(|| format!("neural I/O artifact is missing group {group_name}"))?;
    for &index in &group.engine_indices {
        let neuron_id = pack.neuron_ids[index as usize];
        channels.push(SensoryRateChannel::new(
            feature,
            neuron_id,
            0.0,
            1.0,
            baseline_rate_hz,
            rate_weight_hz_per_unit,
        )?);
        sensory_ids.push(neuron_id);
        sensory_indices.push(index);
        sensory_classes.push(class);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_flight_state_channels(
    pack: &ConnectomePack,
    resolution: &NeuralIoResolution,
    group_name: &str,
    input_max_rad_s: f64,
    input_rate_hz: f64,
    channels: &mut Vec<SensoryRateChannel>,
    sensory_ids: &mut Vec<u64>,
    sensory_indices: &mut Vec<u32>,
    sensory_classes: &mut Vec<SensoryClass>,
) -> Result<()> {
    let group = resolution
        .group(group_name)
        .with_context(|| format!("neural I/O artifact is missing group {group_name}"))?;
    for &index in &group.engine_indices {
        let neuron_id = pack.neuron_ids[index as usize];
        channels.push(SensoryRateChannel::new(
            SensoryFeature::FlightAngularSpeed,
            neuron_id,
            0.0,
            input_max_rad_s,
            0.0,
            input_rate_hz,
        )?);
        sensory_ids.push(neuron_id);
        sensory_indices.push(index);
        sensory_classes.push(SensoryClass::FlightState);
    }
    Ok(())
}

fn collect_bilateral_probe_positions(
    resolution: Option<&NeuralIoResolution>,
    prefix: &str,
    excluded_prefixes: &[&str],
    probe_indices: &mut Vec<u32>,
    position_by_index: &mut BTreeMap<u32, usize>,
) -> (Vec<usize>, Vec<usize>) {
    let Some(resolution) = resolution else {
        return (Vec::new(), Vec::new());
    };
    let mut left = BTreeSet::new();
    let mut right = BTreeSet::new();
    for (name, group) in &resolution.groups {
        if !is_output_probe_group(name, prefix, excluded_prefixes) {
            continue;
        }
        let side = if name.ends_with("_left") {
            &mut left
        } else if name.ends_with("_right") {
            &mut right
        } else {
            continue;
        };
        for &index in &group.engine_indices {
            let position = *position_by_index.entry(index).or_insert_with(|| {
                let position = probe_indices.len();
                probe_indices.push(index);
                position
            });
            side.insert(position);
        }
    }
    (left.into_iter().collect(), right.into_iter().collect())
}

fn collect_probe_positions(
    resolution: Option<&NeuralIoResolution>,
    prefix: &str,
    probe_indices: &mut Vec<u32>,
    position_by_index: &mut BTreeMap<u32, usize>,
) -> Vec<usize> {
    let Some(resolution) = resolution else {
        return Vec::new();
    };
    let mut positions = BTreeSet::new();
    for (name, group) in &resolution.groups {
        if !name.starts_with(prefix) {
            continue;
        }
        for &index in &group.engine_indices {
            let position = *position_by_index.entry(index).or_insert_with(|| {
                let position = probe_indices.len();
                probe_indices.push(index);
                position
            });
            positions.insert(position);
        }
    }
    positions.into_iter().collect()
}

fn is_output_probe_group(name: &str, prefix: &str, excluded_prefixes: &[&str]) -> bool {
    name.starts_with(prefix)
        && !excluded_prefixes
            .iter()
            .any(|excluded| name.starts_with(excluded))
}

fn probe_population_rate(deltas: &[u32], positions: &[usize], window_ms: f64) -> f64 {
    if positions.is_empty() {
        return 0.0;
    }
    positions
        .iter()
        .map(|&position| f64::from(deltas[position]))
        .sum::<f64>()
        * 1000.0
        / (window_ms * positions.len() as f64)
}

fn update_filtered_rate(filtered: &mut f64, current: f64, alpha: f64) {
    *filtered += alpha * (current - *filtered);
}

fn bilateral_decoder(
    left_rate_hz: f64,
    right_rate_hz: f64,
    reference_rate_hz: f64,
    gain: f64,
) -> f64 {
    (gain * (right_rate_hz - left_rate_hz)
        / (left_rate_hz + right_rate_hz + reference_rate_hz * 0.25))
        .clamp(-1.0, 1.0)
}

fn antagonistic_decoder(
    increase_rate_hz: f64,
    decrease_rate_hz: f64,
    reference_rate_hz: f64,
    gain: f64,
) -> f64 {
    (gain * (increase_rate_hz - decrease_rate_hz)
        / (increase_rate_hz + decrease_rate_hz + reference_rate_hz * 0.25))
        .clamp(-1.0, 1.0)
}

fn normalized_population_drive(rate_hz: f64, reference_rate_hz: f64) -> f64 {
    (rate_hz / reference_rate_hz).clamp(0.0, 1.0)
}

fn cns_motor_activation(rate_hz: f64, reference_rate_hz: f64) -> f64 {
    rate_hz / (rate_hz + reference_rate_hz)
}

fn class_index(class: SensoryClass) -> usize {
    match class {
        SensoryClass::Taste => 0,
        SensoryClass::Olfactory => 1,
        SensoryClass::Visual => 2,
        SensoryClass::FlightState => 3,
    }
}

fn contextual_feeding_spikes(male_cns: bool, taste_valence: f64, mn9_spikes: u32) -> u32 {
    if male_cns && taste_valence <= 0.0 {
        0
    } else {
        mn9_spikes
    }
}

fn next_feeding_extension(
    previous: f64,
    mn9_spikes: u32,
    window_ms: f64,
    release_tau_ms: f64,
    extension_per_spike: f64,
) -> f64 {
    let decayed = previous * (-window_ms / release_tau_ms).exp();
    (decayed + f64::from(mn9_spikes) * extension_per_spike).clamp(0.0, 1.0)
}

fn neuron_index(pack: &ConnectomePack, neuron_id: u64) -> Option<u32> {
    pack.neuron_ids
        .iter()
        .position(|&candidate| candidate == neuron_id)
        .map(|index| index as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_io::NeuralIoArtifact;

    #[test]
    fn cns_motor_disconnect_preserves_rates_but_removes_commands() {
        let mut probes = CnsMotorProbes {
            positions: std::array::from_fn(|_| [vec![0], vec![1]]),
            unique_positions: vec![0, 1],
            filtered_rates: [[0.0; 2]; 4],
        };
        let parameters = BrainBridgeParameters::default();
        let connected = probes.update(&[2, 4], 100.0, 1.0, parameters);
        assert_eq!(connected.flight_power_hz, [20.0, 40.0]);
        assert_eq!(connected.spike_delta, 6);
        assert_eq!(connected.flight_activation, 0.6);
        assert!(connected.steering > 0.0);
        let disconnected = probes.update(
            &[2, 4],
            100.0,
            1.0,
            BrainBridgeParameters {
                cns_motor_outputs_enabled: false,
                ..parameters
            },
        );
        assert_eq!(disconnected.flight_power_hz, connected.flight_power_hz);
        assert_eq!(disconnected.flight_activation, 0.0);
        assert_eq!(disconnected.walking_activation, 0.0);
        assert_eq!(disconnected.steering, 0.0);
    }

    #[test]
    fn cns_motor_decoder_has_no_tonic_drive_without_spikes() {
        let mut probes = CnsMotorProbes {
            positions: std::array::from_fn(|_| [vec![0], vec![1]]),
            unique_positions: vec![0, 1],
            filtered_rates: [[0.0; 2]; 4],
        };
        let readout = probes.update(&[0, 0], 2.0, 1.0, BrainBridgeParameters::default());
        assert_eq!(readout.flight_activation, 0.0);
        assert_eq!(readout.walking_activation, 0.0);
        assert_eq!(readout.steering, 0.0);
    }

    #[test]
    fn pinned_food_olfaction_keeps_all_orns_and_maps_profiled_rates() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let artifact =
            NeuralIoArtifact::load(root.join("assets/neuromechfly/flywire_v783_neural_io.json"))
                .unwrap();
        let pack = ConnectomePack::open(root.join("outputs/packs/flywire_v783")).unwrap();
        let resolution = artifact.resolve(&pack).unwrap();
        let bindings = food_olfaction_bindings(&resolution).unwrap();
        assert_eq!(bindings.len(), 401);

        let mut channels = Vec::new();
        let mut sensory_ids = Vec::new();
        let mut sensory_indices = Vec::new();
        let mut sensory_classes = Vec::new();
        append_olfactory_channels(
            &pack,
            &resolution,
            "olfaction_left",
            AntennaSide::Left,
            &bindings,
            ORN_SPONTANEOUS_RATE_HZ,
            OLFACTORY_INPUT_RATE_HZ,
            SensoryClass::Olfactory,
            &mut channels,
            &mut sensory_ids,
            &mut sensory_indices,
            &mut sensory_classes,
        )
        .unwrap();
        append_olfactory_channels(
            &pack,
            &resolution,
            "olfaction_right",
            AntennaSide::Right,
            &bindings,
            ORN_SPONTANEOUS_RATE_HZ,
            OLFACTORY_INPUT_RATE_HZ,
            SensoryClass::Olfactory,
            &mut channels,
            &mut sensory_ids,
            &mut sensory_indices,
            &mut sensory_classes,
        )
        .unwrap();

        assert_eq!(channels.len(), 2_090);
        assert_eq!(sensory_ids.len(), 2_090);
        assert_eq!(
            channels
                .iter()
                .filter(|channel| matches!(channel.feature, SensoryFeature::FoodOdor { .. }))
                .count(),
            401
        );
        assert_eq!(
            channels
                .iter()
                .filter(|channel| {
                    matches!(
                        channel.feature,
                        SensoryFeature::OdorLeft | SensoryFeature::OdorRight
                    )
                })
                .count(),
            1_689
        );
        assert!(channels.iter().all(|channel| {
            channel.baseline_rate_hz == ORN_SPONTANEOUS_RATE_HZ
                && match channel.feature {
                    SensoryFeature::FoodOdor { .. } => {
                        channel.rate_weight_hz_per_unit == OLFACTORY_INPUT_RATE_HZ
                    }
                    SensoryFeature::OdorLeft | SensoryFeature::OdorRight => {
                        channel.rate_weight_hz_per_unit == 0.0
                    }
                    _ => false,
                }
        }));
    }

    #[test]
    fn full_neural_io_fails_closed_without_food_olfaction_profiles() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let artifact =
            NeuralIoArtifact::load(root.join("assets/neuromechfly/flywire_v783_neural_io.json"))
                .unwrap();
        let pack = ConnectomePack::open(root.join("outputs/packs/flywire_v783")).unwrap();
        let mut resolution = artifact.resolve(&pack).unwrap();
        resolution.artifact.food_olfaction = None;

        let error =
            match BrainBodyBridge::build(&pack, Some(resolution), BrainBridgeParameters::default())
            {
                Ok(_) => {
                    panic!("full neural I/O construction should require food-olfaction profiles")
                }
                Err(error) => error.to_string(),
            };
        assert!(error.contains("missing food-olfaction profiles"));
    }

    #[test]
    fn cns_feeding_requires_both_taste_context_and_mn9_spikes() {
        assert_eq!(contextual_feeding_spikes(true, 0.0, 100), 0);
        assert_eq!(contextual_feeding_spikes(true, -1.0, 100), 0);
        assert_eq!(contextual_feeding_spikes(true, 1.0, 0), 0);
        assert_eq!(contextual_feeding_spikes(true, 1.0, 2), 2);
        assert_eq!(contextual_feeding_spikes(false, 0.0, 2), 2);
        let mut extension = 1.0;
        for _ in 0..1000 {
            extension = next_feeding_extension(
                extension,
                contextual_feeding_spikes(true, 0.0, 100),
                2.0,
                500.0,
                0.35,
            );
        }
        assert!(extension < 0.02);
    }

    #[test]
    fn mn9_spikes_drive_a_bounded_leaky_rostrum_command() {
        let first = next_feeding_extension(0.0, 1, 2.0, 500.0, 0.35);
        let second = next_feeding_extension(first, 1, 2.0, 500.0, 0.35);
        let saturated = next_feeding_extension(second, 10, 2.0, 500.0, 0.35);
        let released = next_feeding_extension(saturated, 0, 500.0, 500.0, 0.35);
        assert_eq!(first, 0.35);
        assert!(second > first);
        assert_eq!(saturated, 1.0);
        assert!((released - (-1.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn bilateral_decoder_is_bounded_and_side_sensitive() {
        assert_eq!(bilateral_decoder(0.0, 0.0, 80.0, 1.0), 0.0);
        assert!(bilateral_decoder(5.0, 80.0, 80.0, 1.0) > 0.0);
        assert!(bilateral_decoder(80.0, 5.0, 80.0, 1.0) < 0.0);
        assert!(bilateral_decoder(0.0, 1.0e9, 80.0, 1.0) <= 1.0);
        assert_eq!(bilateral_decoder(5.0, 80.0, 80.0, 0.0), 0.0);
    }

    #[test]
    fn antagonistic_decoder_is_bounded_and_sign_sensitive() {
        assert_eq!(antagonistic_decoder(0.0, 0.0, 80.0, 1.0), 0.0);
        assert!(antagonistic_decoder(80.0, 5.0, 80.0, 1.0) > 0.0);
        assert!(antagonistic_decoder(5.0, 80.0, 80.0, 1.0) < 0.0);
        assert!(antagonistic_decoder(1.0e9, 0.0, 80.0, 1.0) <= 1.0);
        assert_eq!(antagonistic_decoder(80.0, 5.0, 80.0, 0.0), 0.0);
    }

    #[test]
    fn landing_drive_is_normalized_and_bounded() {
        assert_eq!(normalized_population_drive(0.0, 80.0), 0.0);
        assert_eq!(normalized_population_drive(40.0, 80.0), 0.5);
        assert_eq!(normalized_population_drive(80.0, 80.0), 1.0);
        assert_eq!(normalized_population_drive(160.0, 80.0), 1.0);
    }

    #[test]
    fn generic_flight_probes_exclude_state_and_altitude_candidate_groups() {
        assert!(is_output_probe_group(
            "flight_dnp20_left",
            "flight_",
            &[
                "flight_state_",
                "flight_jo_e_",
                "flight_dng02_",
                "flight_dng07_"
            ]
        ));
        assert!(!is_output_probe_group(
            "flight_state_msahn_left",
            "flight_",
            &[
                "flight_state_",
                "flight_jo_e_",
                "flight_dng02_",
                "flight_dng07_"
            ]
        ));
        assert!(!is_output_probe_group(
            "flight_dng02_left",
            "flight_",
            &[
                "flight_state_",
                "flight_jo_e_",
                "flight_dng02_",
                "flight_dng07_"
            ]
        ));
        assert!(!is_output_probe_group(
            "flight_dng07_left",
            "flight_",
            &[
                "flight_state_",
                "flight_jo_e_",
                "flight_dng02_",
                "flight_dng07_"
            ]
        ));
        assert!(!is_output_probe_group(
            "flight_jo_e_left",
            "flight_",
            &[
                "flight_state_",
                "flight_jo_e_",
                "flight_dng02_",
                "flight_dng07_"
            ]
        ));
    }
}
