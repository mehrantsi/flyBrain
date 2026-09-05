use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use mujoco_rs::prelude::MjtObj;
use serde::{Deserialize, Serialize};

use crate::behavior::{BehaviorInput, BehaviorMode, BehaviorParameters, ExplorerController};
use crate::brain_bridge::{
    BrainBodyBridge, BrainBridgeParameters, CnsMotorReadout, MALE_CNS_NEURAL_IO_FILE,
    NEURAL_IO_FILE, NeuralIoStats,
};
use crate::flight::{
    FlightCommand, FlightDynamicsParameters, FlightRuntime, FlightStabilizer, WallLandingTarget,
};
use crate::flight_behavior::{
    FlightBehaviorController, FlightBehaviorInput, FlightBehaviorParameters, FlightMode,
};
use crate::foraging::{
    CnsForagingParameters, ForagingCommand, ForagingController, ForagingInput, ForagingMode,
};
use crate::gait::GaitLibrary;
use crate::grooming::{GroomingController, GroomingInput, GroomingMode, GroomingTrigger};
use crate::habitat::Habitat;
use crate::neural_io::MALE_CNS_MATERIALIZATION;
use crate::obstacle_avoidance::{
    NavigationObservation, NavigationPolicy, NavigationPolicyParameters,
};
use crate::olfaction::OlfactoryTransducer;
use crate::pack::ConnectomePack;
use crate::retina::RetinaSummary;
use crate::world::{MuJoCoWorld, ObstacleSample, WorldEnvironment};

const OVERHANG_DESCENT_MM: f64 = 14.0;
const PLANAR_WALL_ESCAPE_CLEARANCE_MM: f64 = 5.0;
const PLANAR_WALL_ESCAPE_RELEASE_MM: f64 = 20.0;
const WALL_ESCAPE_RELEASE_ALIGNMENT: f64 = 0.9;
const WALL_ESCAPE_RELEASE_INWARD_SPEED_MM_S: f64 = 25.0;
const WALL_ESCAPE_ACCELERATION_ALIGNMENT: f64 = 0.25;
const WALL_ESCAPE_RELEASE_DWELL_WINDOWS: usize = 25;
const WALL_ESCAPE_DEPARTURE_HOLD_WINDOWS: usize = 250;
const GROUNDED_OBSTACLE_SLOWING_DISTANCE_MM: f64 = 55.0;
const GROUNDED_OBSTACLE_FORWARD_GAIN: f64 = 0.55;
const GROUNDED_COLLISION_FORWARD_GAIN: f64 = 0.08;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct SimulationParameters {
    pub brain: BrainBridgeParameters,
    pub behavior: BehaviorParameters,
    pub flight_behavior: FlightBehaviorParameters,
    pub flight_dynamics: FlightDynamicsParameters,
    pub brain_walking_steering_gain: f64,
    #[serde(default)]
    pub cns_foraging: CnsForagingParameters,
    #[serde(default)]
    pub odor_guidance: crate::odor_guidance::OdorGuidanceParameters,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SimulationParameterArtifact {
    pub schema: String,
    pub schema_version: u32,
    pub profile_id: String,
    pub status: String,
    pub topology_sha256: Option<String>,
    pub source_dataset_sha256: Option<String>,
    pub parameters: SimulationParameters,
}

impl SimulationParameterArtifact {
    pub fn validate(&self) -> Result<()> {
        if self.schema != "flybrain.simulation-parameters" || self.schema_version != 1 {
            bail!("unsupported simulation parameter artifact schema")
        }
        if self.profile_id.trim().is_empty() || self.status.trim().is_empty() {
            bail!("simulation parameter profile_id and status must not be empty")
        }
        for (name, digest) in [
            ("topology_sha256", self.topology_sha256.as_deref()),
            (
                "source_dataset_sha256",
                self.source_dataset_sha256.as_deref(),
            ),
        ] {
            if digest.is_some_and(|value| {
                value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
            }) {
                bail!("simulation parameter {name} must be a SHA-256 digest")
            }
        }
        self.parameters.validate()?;
        Ok(())
    }
}

impl Default for SimulationParameters {
    fn default() -> Self {
        Self {
            brain: BrainBridgeParameters::default(),
            behavior: BehaviorParameters::default(),
            flight_behavior: FlightBehaviorParameters::default(),
            flight_dynamics: FlightDynamicsParameters::default(),
            brain_walking_steering_gain: 0.35,
            cns_foraging: CnsForagingParameters::default(),
            odor_guidance: crate::odor_guidance::OdorGuidanceParameters::default(),
        }
    }
}

impl SimulationParameters {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("reading simulation parameters {}", path.display()))?;
        let artifact: SimulationParameterArtifact = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing simulation parameters {}", path.display()))?;
        artifact.validate()?;
        Ok(artifact.parameters)
    }

    pub fn validate(self) -> Result<Self> {
        self.brain.validate()?;
        self.behavior.validate()?;
        self.flight_behavior.validate()?;
        self.flight_dynamics.validate()?;
        self.cns_foraging.validate()?;
        self.odor_guidance.validate()?;
        if !self.brain_walking_steering_gain.is_finite() || self.brain_walking_steering_gain < 0.0 {
            bail!("brain walking steering gain must be finite and non-negative")
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SimulationSnapshot {
    pub time_seconds: f64,
    pub root_position: [f64; 3],
    pub horizontal_speed_mm_s: f64,
    pub forward_speed_mm_s: f64,
    pub body_pitch_deg: f64,
    pub food_center: [f64; 3],
    pub food_distance: f64,
    pub food_enabled: bool,
    pub odor_intensity: f64,
    pub odor_left: f64,
    pub odor_right: f64,
    pub odor_left_ppm: f64,
    pub odor_right_ppm: f64,
    pub visual_left: f64,
    pub visual_right: f64,
    pub visual_contrast_left: f64,
    pub visual_contrast_right: f64,
    pub taste_active: bool,
    pub tasted_resource: Option<usize>,
    pub nearest_resource: Option<usize>,
    pub nearest_resource_distance: f64,
    pub behavior_mode: BehaviorMode,
    pub grooming_mode: GroomingMode,
    pub grooming_trigger: GroomingTrigger,
    pub grooming_active: bool,
    pub grooming_phase: f64,
    pub grooming_support_leg_count: usize,
    pub contact_count: usize,
    pub wall_support_leg_count: usize,
    pub perched_on_wall: bool,
    pub wall_landing_target: Option<WallLandingTarget>,
    pub wall_landing_alignment: Option<f64>,
    pub mn9_spike_delta: u32,
    pub filtered_mn9_rate_hz: f64,
    pub population_spike_delta: u64,
    pub cumulative_spiking_neuron_count: usize,
    pub filtered_population_rate_hz: f64,
    pub brain_field_potential_mv: f64,
    pub brain_field_dominant_frequency_hz: f64,
    pub brain_field_sample_sequence: u64,
    pub taste_event_delta: u64,
    pub olfactory_event_delta: u64,
    pub visual_event_delta: u64,
    pub flight_state_event_delta: u64,
    pub flight_mechanosensory: f64,
    pub walking_dn_left_rate_hz: f64,
    pub walking_dn_right_rate_hz: f64,
    pub flight_dn_left_rate_hz: f64,
    pub flight_dn_right_rate_hz: f64,
    pub flight_power_increase_rate_hz: f64,
    pub flight_power_decrease_rate_hz: f64,
    pub landing_dn_rate_hz: f64,
    pub brain_walking_drive: f64,
    pub brain_walking_steering: f64,
    pub brain_flight_drive: f64,
    pub brain_flight_steering: f64,
    pub brain_altitude_control: f64,
    pub brain_landing_drive: f64,
    pub cns_motor: Option<CnsMotorReadout>,
    pub cns_olfactory: Option<crate::cns_olfaction::CnsOlfactoryReadout>,
    pub odor_guidance: crate::odor_guidance::OdorGuidanceCommand,
    pub foraging_mode: ForagingMode,
    pub flight_allowed: bool,
    pub flight_mode: FlightMode,
    pub flight_amplitude_scale: f64,
    pub flight_frequency_scale: f64,
    pub flight_horizontal_speed_scale: f64,
    pub flight_steering: f64,
    pub flight_odor_steering: f64,
    pub flight_wander_steering: f64,
    pub flight_brain_steering_contribution: f64,
    pub flight_boundary_avoidance: f64,
    pub flight_obstacle_avoidance: f64,
    pub flight_escape_active: bool,
    pub flight_forward_clearance_mm: f64,
    pub flight_up_clearance_mm: f64,
    pub flight_down_clearance_mm: f64,
    pub flight_nearest_obstacle_geom_id: Option<usize>,
    pub environment_contact_count: usize,
    pub ventral_optic_flow_rad_s: f64,
    pub optic_flow_altitude_control: f64,
    pub neural_altitude_contribution_mm_s: f64,
    pub optic_flow_altitude_contribution_mm_s: f64,
    pub flight_altitude_hold: bool,
    pub flight_target_height_mm: f64,
    pub flight_desired_height_mm: f64,
    pub flight_altitude_bounds_mm: [f64; 2],
    pub flight_altitude_target_clamped: bool,
    pub flight_vertical_force_to_weight: f64,
    pub flight_peak_strip_speed_mm_s: f64,
    pub feeding_extension: f64,
    pub forward_gain: f64,
    pub walking_turn_gain: f64,
    pub walking_translation_scale: f64,
    pub brain_wall_seconds: f64,
    pub brain_encoding_seconds: f64,
    pub brain_engine_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct WallEscapeState {
    direction_xy: [f64; 2],
    release_ready_windows: usize,
    release_hold_windows: usize,
    latched: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct WallEscapeObservation {
    mode: FlightMode,
    wall_takeoff: bool,
    wall_support_leg_count: usize,
    position_mm: [f64; 3],
    room_half_extents_mm: [f64; 3],
    forward_xy: [f64; 2],
    planar_velocity_mm_s: [f64; 2],
}

impl Default for WallEscapeState {
    fn default() -> Self {
        Self {
            direction_xy: [1.0, 0.0],
            release_ready_windows: 0,
            release_hold_windows: 0,
            latched: false,
        }
    }
}

impl WallEscapeState {
    fn update(&mut self, observation: WallEscapeObservation) {
        let wall_clearance =
            planar_wall_clearance(observation.position_mm, observation.room_half_extents_mm);
        let airborne_wall_contact = observation.mode != FlightMode::Grounded
            && wall_clearance <= PLANAR_WALL_ESCAPE_CLEARANCE_MM;
        if observation.mode == FlightMode::Grounded {
            self.latched = false;
            self.release_ready_windows = 0;
            self.release_hold_windows = 0;
        } else if !self.latched
            && self.release_hold_windows == 0
            && (observation.wall_takeoff || airborne_wall_contact)
        {
            self.direction_xy =
                wall_inward_direction(observation.position_mm, observation.room_half_extents_mm);
            self.release_ready_windows = 0;
            self.latched = true;
        } else if self.latched {
            let inward =
                wall_inward_direction(observation.position_mm, observation.room_half_extents_mm);
            if wall_clearance <= PLANAR_WALL_ESCAPE_CLEARANCE_MM
                && dot_xy(self.direction_xy, inward) < 0.5
            {
                self.direction_xy = inward;
                self.release_ready_windows = 0;
            }
            let physically_clear = wall_clearance >= PLANAR_WALL_ESCAPE_RELEASE_MM
                && observation.wall_support_leg_count == 0;
            let departing_head_first = dot_xy(observation.forward_xy, self.direction_xy)
                >= WALL_ESCAPE_RELEASE_ALIGNMENT
                && dot_xy(observation.planar_velocity_mm_s, self.direction_xy)
                    >= WALL_ESCAPE_RELEASE_INWARD_SPEED_MM_S;
            if physically_clear && departing_head_first {
                self.release_ready_windows = self.release_ready_windows.saturating_add(1);
                if self.release_ready_windows >= WALL_ESCAPE_RELEASE_DWELL_WINDOWS {
                    self.latched = false;
                    self.release_ready_windows = 0;
                    self.release_hold_windows = WALL_ESCAPE_DEPARTURE_HOLD_WINDOWS;
                }
            } else if physically_clear {
                self.release_ready_windows = self.release_ready_windows.saturating_sub(1);
            } else {
                self.release_ready_windows = 0;
            }
        }
        if !self.latched && self.release_hold_windows > 0 {
            self.release_hold_windows -= 1;
        }
    }
}

pub struct SimulationStepper {
    parameters: SimulationParameters,
    world: MuJoCoWorld,
    gait: GaitLibrary,
    habitat: Habitat,
    olfactory_transducer: OlfactoryTransducer,
    explorer: ExplorerController,
    flight: FlightRuntime,
    flight_stabilizer: FlightStabilizer,
    flight_behavior: FlightBehaviorController,
    foraging: ForagingController,
    odor_guidance: crate::odor_guidance::OdorGuidance,
    navigation: NavigationPolicy,
    ground_navigation: NavigationPolicy,
    obstacle_sample: ObstacleSample,
    obstacle_sample_elapsed_seconds: f64,
    wall_escape: WallEscapeState,
    grooming: GroomingController,
    brain: Option<BrainBodyBridge>,
    brain_telemetry_enabled: bool,
    pack_path: Option<PathBuf>,
    neural_io_path: PathBuf,
    brain_materialization: Option<String>,
    control_steps: usize,
    settle_seconds: f64,
    phase_rad: f64,
    forward_gain: f64,
    turn_gain: f64,
    walking_translation_scale: f64,
    landing_target_mm: Option<f64>,
    wall_landing: Option<WallLandingTarget>,
    touchdown_gait_ramp: f64,
    standing_joint_controls: [f64; 42],
    feeding_pose_held: bool,
    feeding_extension: f64,
    food_center: [f64; 3],
    food_enabled: bool,
    flight_allowed: bool,
    retina_summaries: [RetinaSummary; 2],
    snapshot: SimulationSnapshot,
}

impl SimulationStepper {
    pub fn new(
        assets: impl AsRef<Path>,
        pack_path: Option<impl AsRef<Path>>,
        control_hz: f64,
        settle_seconds: f64,
    ) -> Result<Self> {
        Self::new_with_parameters(
            assets,
            pack_path,
            control_hz,
            settle_seconds,
            SimulationParameters::default(),
        )
    }

    pub fn new_with_parameters(
        assets: impl AsRef<Path>,
        pack_path: Option<impl AsRef<Path>>,
        control_hz: f64,
        settle_seconds: f64,
        parameters: SimulationParameters,
    ) -> Result<Self> {
        let parameters = parameters.validate()?;
        if !control_hz.is_finite() || control_hz <= 0.0 {
            bail!("control_hz must be finite and positive")
        }
        if !settle_seconds.is_finite() || settle_seconds < 0.0 {
            bail!("settle_seconds must be finite and non-negative")
        }
        let assets = assets.as_ref();
        let mut world = MuJoCoWorld::from_assets_dir(assets)?;
        let gait = GaitLibrary::open(assets.join("tripod_gait.json"))?;
        let habitat = Habitat::load(assets)?;
        let flight =
            FlightRuntime::new_with_parameters(assets, &world, parameters.flight_dynamics)?;
        let obstacle_sample = world.obstacle_sample(180.0)?;
        for resource in habitat.resources() {
            if world
                .model()
                .name_to_id(MjtObj::mjOBJ_GEOM, &resource.geom)
                .is_none()
            {
                bail!(
                    "habitat resource {} is missing geom {}",
                    resource.id,
                    resource.geom
                )
            }
        }
        let timestep = world.timestep_seconds();
        let brain_timestep = parameters.brain.neural.dt_ms / 1000.0;
        if (timestep - brain_timestep).abs() > 1e-12 {
            bail!("MuJoCo and brain timesteps do not match")
        }
        let control_steps = rounded_positive_ratio(1.0 / control_hz, timestep)?;
        let pack_path = pack_path.map(|path| path.as_ref().to_path_buf());
        let neural_io_path = assets.join(NEURAL_IO_FILE);
        let (brain, brain_materialization) =
            load_brain(pack_path.as_deref(), &neural_io_path, parameters.brain)?;
        let food_center = world.metadata().environment.food_center;
        let root_position = world.root_position();
        let flight_altitude_bounds_mm = habitat.room().flight_altitude_bounds_mm;
        let standing_joint_controls = std::array::from_fn(|index| world.neutral_control()[index]);
        let mut stepper = Self {
            parameters,
            world,
            gait,
            habitat,
            olfactory_transducer: OlfactoryTransducer::default(),
            explorer: ExplorerController::with_parameters(
                0x5eed_f17b_2026_0816,
                parameters.behavior,
            )?,
            flight,
            flight_stabilizer: FlightStabilizer::from_dynamics(parameters.flight_dynamics)?,
            flight_behavior: FlightBehaviorController::with_parameters(
                0xa17f_1eaf_2026_0816,
                parameters.flight_behavior,
            )?,
            foraging: ForagingController::default(),
            odor_guidance: crate::odor_guidance::OdorGuidance::default(),
            navigation: NavigationPolicy::default(),
            ground_navigation: NavigationPolicy::with_parameters(NavigationPolicyParameters {
                obstacle_trigger_mm: 2.0,
                obstacle_release_mm: 4.0,
                escape_clear_dwell_seconds: 0.1,
                ..Default::default()
            })?,
            obstacle_sample,
            obstacle_sample_elapsed_seconds: 0.0,
            wall_escape: WallEscapeState::default(),
            grooming: GroomingController::new(),
            brain,
            brain_telemetry_enabled: false,
            pack_path,
            neural_io_path,
            brain_materialization,
            control_steps,
            settle_seconds,
            phase_rad: 0.0,
            forward_gain: 1.0,
            turn_gain: 0.0,
            walking_translation_scale: 1.0,
            landing_target_mm: None,
            wall_landing: None,
            touchdown_gait_ramp: 1.0,
            standing_joint_controls,
            feeding_pose_held: false,
            feeding_extension: 0.0,
            food_center,
            food_enabled: true,
            flight_allowed: true,
            retina_summaries: [RetinaSummary::default(); 2],
            snapshot: SimulationSnapshot {
                root_position,
                food_center,
                food_enabled: true,
                flight_allowed: true,
                flight_altitude_bounds_mm,
                forward_gain: 1.0,
                ..SimulationSnapshot::default()
            },
        };
        stepper.refresh_environment_snapshot()?;
        Ok(stepper)
    }

    pub fn step_window(&mut self) -> Result<SimulationSnapshot> {
        self.step_window_steps(self.control_steps)
    }

    pub fn step_window_steps(&mut self, window_steps: usize) -> Result<SimulationSnapshot> {
        if window_steps == 0 {
            bail!("simulation window must contain at least one physics step")
        }
        let window_seconds = window_steps as f64 * self.world.timestep_seconds();
        let was_airborne = self.snapshot.flight_mode != FlightMode::Grounded;
        let mut sample = self.world.sensory_sample()?;
        if was_airborne || (!self.feeding_pose_held && self.touchdown_gait_ramp >= 1.0) {
            self.standing_joint_controls =
                std::array::from_fn(|index| self.world.controls()[index]);
        }
        self.feeding_pose_held =
            self.snapshot.behavior_mode == BehaviorMode::Feed || self.feeding_extension > 0.01;
        if was_airborne {
            self.touchdown_gait_ramp = 0.0;
        } else {
            self.touchdown_gait_ramp = (self.touchdown_gait_ramp + window_seconds / 0.16).min(1.0);
        }
        let ramp = if self.settle_seconds == 0.0 {
            1.0
        } else {
            (self.world.time() / self.settle_seconds).clamp(0.0, 1.0)
        } * ((self.touchdown_gait_ramp - 0.25) / 0.75).clamp(0.0, 1.0);
        let grooming_preparing = self.grooming.preparing();
        let cns_gait = self
            .brain
            .as_ref()
            .is_some_and(BrainBodyBridge::is_male_cns);
        let excursion = if grooming_preparing {
            0.0
        } else {
            gait_excursion_gain(self.forward_gain, self.feeding_extension, cns_gait)
        };
        let gait_command = if cns_gait {
            self.gait.sample_bilateral(
                self.phase_rad,
                walking_side_drive(1.0, self.walking_translation_scale, self.turn_gain),
            )?
        } else {
            self.gait
                .sample(self.phase_rad, excursion, self.turn_gain)?
        };
        let mut joint_controls = [0.0; 42];
        for (index, control) in joint_controls.iter_mut().enumerate() {
            let target = if cns_gait {
                self.standing_joint_controls[index] * (1.0 - excursion)
                    + gait_command.joint_controls[index] * excursion
            } else if self.forward_gain == 0.0 && self.feeding_pose_held {
                self.standing_joint_controls[index]
            } else {
                gait_command.joint_controls[index]
            };
            *control = if was_airborne {
                self.world.neutral_control()[index]
            } else {
                self.standing_joint_controls[index] * (1.0 - ramp) + target * ramp
            };
        }
        let mut adhesion = if !was_airborne && ramp > 0.0 {
            gait_command.adhesion
        } else {
            [0.0; 6]
        };
        if !was_airborne && ramp == 0.0 {
            adhesion = sample.foot_contacts.map(f64::from);
        } else if cns_gait {
            adhesion
                .iter_mut()
                .for_each(|control| *control *= excursion.min(1.0));
        } else if self.forward_gain == 0.0 && self.feeding_pose_held {
            adhesion = [0.0; 6];
        }
        if !was_airborne && self.feeding_pose_held && (!cns_gait || self.feeding_extension > 0.1) {
            adhesion = sample.foot_contacts.map(f64::from);
        }
        clamp_joint_controls_to_actuator_ranges(&self.world, &mut joint_controls)?;
        self.world.set_joint_controls(&joint_controls)?;
        self.world.set_adhesion_controls(&adhesion)?;
        self.world.set_feeding_extension(self.feeding_extension)?;

        let environment = self.world.metadata().environment.clone();
        let root_position = self.world.root_position();
        let root_quaternion = self.world.root_quaternion();
        let root_velocity = self.world.root_velocity();
        let horizontal_speed_mm_s = root_velocity[3].hypot(root_velocity[4]);
        self.obstacle_sample_elapsed_seconds += window_seconds;
        if self.obstacle_sample_elapsed_seconds + f64::EPSILON >= 0.01 {
            self.obstacle_sample = self.world.obstacle_sample(180.0)?;
            self.obstacle_sample_elapsed_seconds %= 0.01;
        }
        let obstacle_sample = self.obstacle_sample;
        let overhead =
            obstacle_sample.overhead_geom_id.is_some() && obstacle_sample.up_clearance_mm <= 55.0;
        let ventral_optic_flow_rad_s =
            horizontal_speed_mm_s / obstacle_sample.down_clearance_mm.max(3.0);
        let optic_flow_altitude_control = optic_flow_altitude_control(
            horizontal_speed_mm_s,
            self.world.root_position()[2],
            obstacle_sample.down_clearance_mm,
            overhead,
        );
        let taste_position = self.world.body_position(&environment.taste_source_body)?;
        let left_antenna = self.world.body_position("fly/l_funiculus")?;
        let right_antenna = self.world.body_position("fly/r_funiculus")?;
        let habitat_sample = self.habitat.sample(
            left_antenna,
            right_antenna,
            taste_position,
            self.food_center,
            self.food_enabled,
        );
        let olfactory_sample = self.olfactory_transducer.update(
            [habitat_sample.odor_left_ppm, habitat_sample.odor_right_ppm],
            window_seconds,
        )?;
        let food_distance = distance(taste_position, self.food_center);
        let odor_intensity = 0.5
            * (olfactory_sample.perceived_intensity[0] + olfactory_sample.perceived_intensity[1]);
        let taste_active = habitat_sample.tasted_resource.is_some();
        sample.odor_intensity = odor_intensity;
        sample.odor_left = olfactory_sample.perceived_intensity[0];
        sample.odor_right = olfactory_sample.perceived_intensity[1];
        sample.food_odor_activation = olfactory_sample.receptor_activation;
        sample.taste_valence = habitat_sample.taste_valence;
        sample.visual_left = self.retina_summaries[0].mean_intensity;
        sample.visual_right = self.retina_summaries[1].mean_intensity;
        sample.visual_contrast_left = self.retina_summaries[0].spatial_contrast;
        sample.visual_contrast_right = self.retina_summaries[1].spatial_contrast;
        let forward_xy = planar_forward(root_quaternion);
        let forward_speed = dot_xy([root_velocity[3], root_velocity[4]], forward_xy);
        let side_speed = dot_xy(
            [root_velocity[3], root_velocity[4]],
            [-forward_xy[1], forward_xy[0]],
        );
        let yaw_speed = root_velocity[2];
        sample.visual_motion = [
            obstacle_sample.left_clearance_mm,
            obstacle_sample.right_clearance_mm,
        ]
        .map(|clearance| {
            ((forward_speed.abs() / clearance.max(3.0) + yaw_speed.abs()) / 20.0).clamp(0.0, 1.0)
        });
        let forward_loom = forward_speed.max(0.0) / obstacle_sample.forward_clearance_mm.max(3.0);
        sample.visual_loom = [
            ((forward_loom + side_speed.max(0.0) / obstacle_sample.left_clearance_mm.max(3.0))
                / 10.0)
                .clamp(0.0, 1.0),
            ((forward_loom + (-side_speed).max(0.0) / obstacle_sample.right_clearance_mm.max(3.0))
                / 10.0)
                .clamp(0.0, 1.0),
        ];
        sample.angular_velocity_rad_s = root_velocity[..3]
            .try_into()
            .expect("root angular velocity has three components");
        sample.flight_angular_speed_rad_s = if self.snapshot.flight_mode == FlightMode::Grounded {
            0.0
        } else {
            sample
                .angular_velocity_rad_s
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        };
        sample.flight_mechanosensory = if self
            .brain
            .as_ref()
            .is_some_and(BrainBodyBridge::is_male_cns)
        {
            let airflow = self.habitat.airflow_mm_s();
            let relative_air_speed = (0..3)
                .map(|axis| (root_velocity[axis + 3] - airflow[axis]).powi(2))
                .sum::<f64>()
                .sqrt();
            (relative_air_speed / 300.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let contact_count = sample.foot_contacts.iter().filter(|&&value| value).count();
        let wall_support_contacts = self.world.wall_foot_contacts();
        let wall_support_leg_count = wall_support_contacts.iter().filter(|&&value| value).count();
        let perched_on_wall = wall_support_leg_count >= 2;
        let behavior = self.explorer.update(BehaviorInput {
            dt_seconds: window_seconds,
            odor_left: olfactory_sample.perceived_intensity[0],
            odor_right: olfactory_sample.perceived_intensity[1],
            taste_valence: if self.snapshot.flight_mode == FlightMode::Grounded {
                habitat_sample.taste_valence
            } else {
                0.0
            },
        })?;
        sample.taste_valence *= behavior.sensory_taste_gain;
        let perceived_taste_active = taste_active && behavior.sensory_taste_gain > 0.0;

        let mut mn9_spike_delta = 0;
        let mut filtered_mn9_rate_hz = 0.0;
        let mut population_spike_delta = 0;
        let mut cumulative_spiking_neuron_count = self.snapshot.cumulative_spiking_neuron_count;
        let mut filtered_population_rate_hz = 0.0;
        let mut brain_field_potential_mv = self.snapshot.brain_field_potential_mv;
        let mut brain_field_dominant_frequency_hz = self.snapshot.brain_field_dominant_frequency_hz;
        let mut brain_field_sample_sequence = self.snapshot.brain_field_sample_sequence;
        let mut taste_event_delta = 0;
        let mut olfactory_event_delta = 0;
        let mut visual_event_delta = 0;
        let mut flight_state_event_delta = 0;
        let mut walking_dn_left_rate_hz = 0.0;
        let mut walking_dn_right_rate_hz = 0.0;
        let mut flight_dn_left_rate_hz = 0.0;
        let mut flight_dn_right_rate_hz = 0.0;
        let mut flight_power_increase_rate_hz = 0.0;
        let mut flight_power_decrease_rate_hz = 0.0;
        let mut landing_dn_rate_hz = 0.0;
        let mut brain_walking_drive = 0.0;
        let mut brain_walking_steering = 0.0;
        let mut brain_flight_drive = 0.0;
        let mut brain_flight_steering = 0.0;
        let mut brain_altitude_control = 0.0;
        let mut brain_landing_drive = 0.0;
        let mut cns_motor = None;
        let mut cns_olfactory = None;
        let mut brain_wall_seconds = 0.0;
        let mut brain_encoding_seconds = 0.0;
        let mut brain_engine_seconds = 0.0;
        let mut next_motor = if let Some(bridge) = self.brain.as_mut() {
            let result = bridge.run_window(&sample, window_steps)?;
            mn9_spike_delta = result.mn9_spike_delta;
            filtered_mn9_rate_hz = result.filtered_mn9_rate_hz;
            population_spike_delta = result.population_spike_delta;
            cumulative_spiking_neuron_count = result.cumulative_spiking_neuron_count;
            filtered_population_rate_hz = result.filtered_population_rate_hz;
            brain_field_potential_mv = result.brain_field_potential_mv;
            brain_field_dominant_frequency_hz = result.brain_field_dominant_frequency_hz;
            brain_field_sample_sequence = result.brain_field_sample_sequence;
            taste_event_delta = result.taste_event_count;
            olfactory_event_delta = result.olfactory_event_count;
            visual_event_delta = result.visual_event_count;
            flight_state_event_delta = result.flight_state_event_count;
            walking_dn_left_rate_hz = result.walking_left_rate_hz;
            walking_dn_right_rate_hz = result.walking_right_rate_hz;
            flight_dn_left_rate_hz = result.flight_left_rate_hz;
            flight_dn_right_rate_hz = result.flight_right_rate_hz;
            flight_power_increase_rate_hz = result.flight_power_increase_rate_hz;
            flight_power_decrease_rate_hz = result.flight_power_decrease_rate_hz;
            landing_dn_rate_hz = result.landing_dn_rate_hz;
            brain_walking_drive = result.brain_walking_drive;
            brain_walking_steering = result.brain_walking_steering;
            brain_flight_drive = result.brain_flight_drive;
            brain_flight_steering = result.brain_flight_steering;
            brain_altitude_control = result.brain_altitude_control;
            brain_landing_drive = result.brain_landing_drive;
            cns_motor = result.cns_motor;
            cns_olfactory = result.cns_olfactory;
            brain_wall_seconds = result.elapsed.as_secs_f64();
            brain_encoding_seconds = result.encoding_elapsed.as_secs_f64();
            brain_engine_seconds = result.engine_elapsed.as_secs_f64();
            let feeding_gate = 1.0 - result.feeding_extension;
            if result.cns_motor.is_some() {
                (
                    result.forward_gain,
                    result.turn_gain,
                    result.feeding_extension,
                )
            } else {
                (
                    full_brain_walking_forward(
                        behavior.mode,
                        behavior.forward_gain,
                        result.brain_walking_drive * feeding_gate,
                    ),
                    full_brain_walking_turn(
                        behavior.mode,
                        behavior.turn_gain,
                        result.brain_walking_steering,
                        self.parameters.brain_walking_steering_gain,
                    ),
                    result.feeding_extension,
                )
            }
        } else {
            (behavior.forward_gain, behavior.turn_gain, 0.0)
        };

        let odor_guidance = self.odor_guidance.update(
            cns_olfactory.unwrap_or_default(),
            root_position[2],
            window_seconds,
            cns_motor.is_some_and(|motor| motor.outputs_connected)
                && behavior.sensory_taste_gain > 0.0,
            self.parameters.odor_guidance,
        );
        if odor_guidance.active && behavior.mode != BehaviorMode::Feed {
            next_motor.1 = odor_guidance.steering;
        }
        let previous_flight_mode = self.snapshot.flight_mode;
        let foraging = if self.brain.is_some() {
            self.foraging.update(ForagingInput {
                dt_seconds: window_seconds,
                brain_enabled: true,
                flight_mode: previous_flight_mode,
                behavior_mode: behavior.mode,
                odor_left: olfactory_sample.perceived_intensity[0],
                odor_right: olfactory_sample.perceived_intensity[1],
                taste_active: perceived_taste_active,
                surface_contact_count: contact_count,
                brain_flight_drive,
                brain_landing_drive,
                cns_calibration: cns_motor.map(|_| self.parameters.cns_foraging),
                cns_odor_guidance: (cns_motor.is_some() && self.parameters.odor_guidance.enabled)
                    .then_some(odor_guidance),
            })?
        } else {
            ForagingCommand::default()
        };
        let food_contact_blocks_flight = grounded_food_contact_blocks_flight(
            previous_flight_mode,
            perceived_taste_active,
            behavior.mode,
        );
        let flight_behavior = self.flight_behavior.update(FlightBehaviorInput {
            dt_seconds: window_seconds,
            enabled: self.flight_allowed && !food_contact_blocks_flight,
            brain_enabled: self.brain.is_some(),
            root_height_mm: self.world.root_position()[2],
            vertical_velocity_mm_s: root_velocity[5],
            angular_speed_rad_s: root_velocity[..3]
                .iter()
                .map(|rate| rate * rate)
                .sum::<f64>()
                .sqrt(),
            contact_count: if self
                .wall_landing
                .is_some_and(|target| {
                    target.alignment(root_quaternion) < 0.9 || wall_support_leg_count < 3
                })
            {
                0
            } else {
                contact_count
            },
            odor_left: olfactory_sample.perceived_intensity[0],
            odor_right: olfactory_sample.perceived_intensity[1],
            taste_valence: sample.taste_valence,
            brain_flight_drive,
            cns_motor_activation: cns_motor.map(|motor| motor.flight_activation),
            brain_steering: if odor_guidance.active {
                odor_guidance.steering * cns_motor.map_or(0.0, |motor| motor.flight_activation)
            } else {
                brain_flight_steering
            },
            brain_altitude_control,
            optic_flow_altitude_control,
            altitude_hold: overhead,
            cns_food_approach: cns_motor.is_some() && foraging.mode == ForagingMode::Approach,
            cns_approach_height_mm: odor_guidance
                .active
                .then_some(odor_guidance.approach_height_mm),
            landing_request: foraging.landing_request,
            takeoff_inhibited: foraging.takeoff_inhibited,
            collision_escape_active: self.wall_escape.latched
                || self.wall_escape.release_hold_windows > 0,
            flight_altitude_bounds_mm: self.habitat.room().flight_altitude_bounds_mm,
        })?;
        match flight_behavior.mode {
            FlightMode::Landing
                if self.wall_landing.is_none()
                    && planar_wall_clearance(
                        root_position,
                        self.habitat.room().half_extents_mm,
                    ) <= PLANAR_WALL_ESCAPE_RELEASE_MM =>
            {
                self.wall_landing = nearby_wall_landing_target(&self.world, 12.0);
            }
            FlightMode::Takeoff | FlightMode::Cruise => self.wall_landing = None,
            _ => {}
        }
        let wall_takeoff = previous_flight_mode == FlightMode::Grounded
            && flight_behavior.mode == FlightMode::Takeoff
            && perched_on_wall;
        let forward_xy = planar_forward(root_quaternion);
        self.wall_escape.update(WallEscapeObservation {
            mode: flight_behavior.mode,
            wall_takeoff,
            wall_support_leg_count,
            position_mm: root_position,
            room_half_extents_mm: self.habitat.room().half_extents_mm,
            forward_xy,
            planar_velocity_mm_s: [root_velocity[3], root_velocity[4]],
        });
        let grounded = flight_behavior.mode == FlightMode::Grounded;
        if grounded != (previous_flight_mode == FlightMode::Grounded) {
            self.navigation.reset();
            self.ground_navigation.reset();
        }
        let navigation_policy = if grounded {
            &mut self.ground_navigation
        } else {
            &mut self.navigation
        };
        let navigation = navigation_policy.update(NavigationObservation {
            position_mm: root_position,
            forward_xy,
            room_half_extents_mm: self.habitat.room().half_extents_mm,
            forward_clearance_mm: obstacle_sample.forward_clearance_mm,
            left_clearance_mm: obstacle_sample.left_clearance_mm,
            right_clearance_mm: obstacle_sample.right_clearance_mm,
            up_clearance_mm: obstacle_sample.up_clearance_mm,
            overhead: !grounded && overhead,
            dt_seconds: window_seconds,
        })?;
        let landing = flight_behavior.mode == FlightMode::Landing;
        let wall_escape_active = !landing
            && flight_behavior.mode != FlightMode::Grounded
            && (self.wall_escape.latched || self.wall_escape.release_hold_windows > 0);
        let navigation_override = navigation.obstacle_active || navigation.boundary_active;
        let mut next_walking_translation_scale = 1.0;
        if flight_behavior.mode == FlightMode::Grounded && navigation_override {
            let translation_scale = grounded_navigation_forward_gain(
                obstacle_sample.forward_clearance_mm,
                navigation.collision_reflex_active,
            );
            if cns_motor.is_some() {
                next_walking_translation_scale = translation_scale;
            } else {
                next_motor.0 *= translation_scale;
            }
            next_motor.1 = navigation.steering;
        }
        let collision_reflex_active =
            !landing && (wall_escape_active || navigation.collision_reflex_active);
        let flight_steering = if landing || wall_escape_active {
            0.0
        } else if navigation.collision_reflex_active {
            navigation.steering
        } else {
            flight_behavior.steering
        };
        let planar_velocity_direction = if collision_reflex_active {
            Some(if wall_escape_active {
                self.wall_escape.direction_xy
            } else {
                navigation.direction_xy
            })
        } else {
            None
        };
        let horizontal_speed_scale = (if landing {
            0.0
        } else if wall_escape_active {
            wall_escape_speed_scale(forward_xy, self.wall_escape.direction_xy)
                * if self.wall_escape.latched { 0.35 } else { 1.0 }
        } else if navigation.collision_reflex_active {
            1.0
        } else {
            foraging.horizontal_speed_scale
        }) * cns_motor
            .map_or(1.0, |motor| motor.flight_activation.sqrt());
        let grooming_command = self.grooming.update(GroomingInput {
            dt_seconds: window_seconds,
            grounded: flight_behavior.mode == FlightMode::Grounded
                && cns_motor.is_none_or(|motor| motor.outputs_connected),
            contact_count,
            allow_fallback: self.brain.is_none(),
            taste_active: perceived_taste_active,
            taste_valence: habitat_sample.taste_valence,
            feeding_extension: next_motor.2,
        })?;
        if flight_behavior.mode == FlightMode::Grounded {
            if was_airborne {
                self.phase_rad = 0.0;
                joint_controls = self.standing_joint_controls;
                adhesion = sample.foot_contacts.map(f64::from);
            }
            if perched_on_wall || grooming_command.active || self.grooming.preparing() {
                joint_controls = std::array::from_fn(|index| self.world.neutral_control()[index]);
                adhesion = [1.0; 6];
            }
            self.grooming.apply(&mut joint_controls, &mut adhesion);
            clamp_joint_controls_to_actuator_ranges(&self.world, &mut joint_controls)?;
            self.world.set_joint_controls(&joint_controls)?;
            self.world.set_adhesion_controls(&adhesion)?;
        }
        if flight_behavior.mode != FlightMode::Grounded {
            let neutral_joint_controls: [f64; 42] =
                std::array::from_fn(|index| self.world.neutral_control()[index]);
            self.world.set_joint_controls(&neutral_joint_controls)?;
            let wall_adhesion = if landing
                && self
                    .wall_landing
                    .is_some_and(|target| target.alignment(root_quaternion) >= 0.9)
            {
                [1.0; 6]
            } else {
                [0.0; 6]
            };
            self.world.set_adhesion_controls(&wall_adhesion)?;
        }
        let desired_height_mm = if let Some(target) = self.wall_landing.filter(|_| landing) {
            target.surface_point_mm[2]
        } else if landing {
            let surface_target = surface_relative_landing_height(
                root_position[2],
                obstacle_sample.down_clearance_mm,
                self.parameters.flight_behavior.landing_height_mm,
            );
            let target = self.landing_target_mm.get_or_insert(root_position[2]);
            *target = (*target - 12.0 * window_seconds).max(surface_target);
            *target
        } else {
            self.landing_target_mm = None;
            flight_behavior.target_height_mm
        };
        let minimum_command_height_mm = if flight_behavior.mode == FlightMode::Landing {
            self.parameters.flight_behavior.landing_height_mm
        } else {
            self.habitat.room().flight_altitude_bounds_mm[0]
        };
        let commanded_height_mm = commanded_flight_height(
            flight_behavior.mode,
            desired_height_mm,
            root_position[2],
            minimum_command_height_mm,
            overhead,
            navigation.altitude_escape,
        );
        let base_flight_command = FlightCommand {
            enabled: flight_behavior.mode != FlightMode::Grounded,
            amplitude: 1.0,
            steering: flight_steering,
            wing_steering_scale: if wall_escape_active { 0.0 } else { 1.0 },
            horizontal_speed_scale,
            heading_target_xy: wall_escape_active.then_some(self.wall_escape.direction_xy),
            planar_velocity_direction,
            altitude_target_mm: (flight_behavior.mode != FlightMode::Grounded)
                .then_some(commanded_height_mm),
            body_pitch_target_rad: (flight_behavior.mode == FlightMode::Landing
                && !perched_on_wall)
                .then_some(0.0),
            wall_landing: self.wall_landing.filter(|_| landing),
            frequency_scale: flight_frequency_scale(
                flight_behavior.mode,
                commanded_height_mm - root_position[2],
                root_velocity[5],
            ),
            pitch_bias_rad: 0.0,
            roll_bias_rad: 0.0,
            differential_pitch_rad: 0.0,
            differential_roll_rad: 0.0,
        };
        let mut flight_vertical_force_to_weight = 0.0;
        let mut flight_peak_strip_speed_mm_s = 0.0_f64;
        for _ in 0..window_steps {
            let mut command_base = base_flight_command;
            if wall_escape_active {
                let forward_xy = planar_forward(self.world.root_quaternion());
                command_base.horizontal_speed_scale =
                    wall_escape_speed_scale(forward_xy, self.wall_escape.direction_xy)
                        * if self.wall_escape.latched { 0.35 } else { 1.0 }
                        * cns_motor.map_or(1.0, |motor| motor.flight_activation.sqrt());
            }
            let command = self.flight_stabilizer.command_with_base_limited(
                self.world.root_quaternion(),
                self.world.root_velocity(),
                command_base,
                flight_behavior.amplitude_scale,
                self.flight.config(),
            )?;
            let telemetry =
                self.flight
                    .advance(&mut self.world, command, self.habitat.airflow_mm_s())?;
            flight_vertical_force_to_weight += telemetry.vertical_force_to_weight;
            flight_peak_strip_speed_mm_s = flight_peak_strip_speed_mm_s.max(
                telemetry
                    .wings
                    .iter()
                    .map(|wing| wing.peak_strip_speed_mm_s)
                    .fold(0.0_f64, f64::max),
            );
        }
        flight_vertical_force_to_weight /= window_steps as f64;
        self.phase_rad = self.gait.advance_phase(
            self.phase_rad,
            window_seconds,
            if flight_behavior.mode != FlightMode::Grounded
                || grooming_command.active
                || self.grooming.preparing()
                || (cns_gait && self.feeding_extension > 0.1)
            {
                0.0
            } else {
                self.forward_gain * ramp
            },
        )?;
        (self.forward_gain, self.turn_gain, self.feeding_extension) = next_motor;
        self.walking_translation_scale = next_walking_translation_scale;
        if grooming_command.active || self.grooming.preparing() {
            self.forward_gain = 0.0;
        }
        self.snapshot = SimulationSnapshot {
            time_seconds: self.world.time(),
            root_position: self.world.root_position(),
            horizontal_speed_mm_s: {
                let velocity = self.world.root_velocity();
                velocity[3].hypot(velocity[4])
            },
            forward_speed_mm_s: planar_forward_speed(&self.world),
            body_pitch_deg: body_pitch_deg(&self.world),
            food_center: self.food_center,
            food_distance,
            food_enabled: self.food_enabled,
            odor_intensity,
            odor_left: olfactory_sample.perceived_intensity[0],
            odor_right: olfactory_sample.perceived_intensity[1],
            odor_left_ppm: olfactory_sample.concentration_ppm[0],
            odor_right_ppm: olfactory_sample.concentration_ppm[1],
            visual_left: self.retina_summaries[0].mean_intensity,
            visual_right: self.retina_summaries[1].mean_intensity,
            visual_contrast_left: self.retina_summaries[0].spatial_contrast,
            visual_contrast_right: self.retina_summaries[1].spatial_contrast,
            taste_active,
            tasted_resource: habitat_sample.tasted_resource,
            nearest_resource: habitat_sample.nearest_resource,
            nearest_resource_distance: habitat_sample.nearest_distance_mm,
            behavior_mode: behavior.mode,
            grooming_mode: grooming_command.mode,
            grooming_trigger: grooming_command.trigger,
            grooming_active: grooming_command.active,
            grooming_phase: grooming_command.phase,
            grooming_support_leg_count: grooming_command.support_leg_count,
            contact_count,
            wall_support_leg_count,
            perched_on_wall,
            wall_landing_target: self.wall_landing,
            wall_landing_alignment: self
                .wall_landing
                .map(|target| target.alignment(self.world.root_quaternion())),
            mn9_spike_delta,
            filtered_mn9_rate_hz,
            population_spike_delta,
            cumulative_spiking_neuron_count,
            filtered_population_rate_hz,
            brain_field_potential_mv,
            brain_field_dominant_frequency_hz,
            brain_field_sample_sequence,
            taste_event_delta,
            olfactory_event_delta,
            visual_event_delta,
            flight_state_event_delta,
            flight_mechanosensory: sample.flight_mechanosensory,
            walking_dn_left_rate_hz,
            walking_dn_right_rate_hz,
            flight_dn_left_rate_hz,
            flight_dn_right_rate_hz,
            flight_power_increase_rate_hz,
            flight_power_decrease_rate_hz,
            landing_dn_rate_hz,
            brain_walking_drive,
            brain_walking_steering,
            brain_flight_drive,
            brain_flight_steering,
            brain_altitude_control,
            brain_landing_drive,
            cns_motor,
            cns_olfactory,
            odor_guidance,
            foraging_mode: foraging.mode,
            flight_allowed: self.flight_allowed,
            flight_mode: flight_behavior.mode,
            flight_amplitude_scale: flight_behavior.amplitude_scale,
            flight_frequency_scale: base_flight_command.frequency_scale,
            flight_horizontal_speed_scale: base_flight_command.horizontal_speed_scale,
            flight_steering,
            flight_odor_steering: flight_behavior.odor_steering_contribution,
            flight_wander_steering: flight_behavior.wander_steering_contribution,
            flight_brain_steering_contribution: flight_behavior.brain_steering_contribution,
            flight_boundary_avoidance: navigation.boundary_steering,
            flight_obstacle_avoidance: navigation.obstacle_steering,
            flight_escape_active: navigation.escape_active || wall_escape_active,
            flight_forward_clearance_mm: obstacle_sample.forward_clearance_mm,
            flight_up_clearance_mm: obstacle_sample.up_clearance_mm,
            flight_down_clearance_mm: obstacle_sample.down_clearance_mm,
            flight_nearest_obstacle_geom_id: obstacle_sample.nearest_geom_id,
            environment_contact_count: obstacle_sample.environment_contact_count,
            ventral_optic_flow_rad_s,
            optic_flow_altitude_control,
            neural_altitude_contribution_mm_s: flight_behavior.neural_altitude_contribution_mm_s,
            optic_flow_altitude_contribution_mm_s: flight_behavior
                .optic_flow_altitude_contribution_mm_s,
            flight_altitude_hold: overhead,
            flight_target_height_mm: commanded_height_mm,
            flight_desired_height_mm: desired_height_mm,
            flight_altitude_bounds_mm: self.habitat.room().flight_altitude_bounds_mm,
            flight_altitude_target_clamped: flight_behavior.altitude_target_clamped,
            flight_vertical_force_to_weight,
            flight_peak_strip_speed_mm_s,
            feeding_extension: self.feeding_extension,
            forward_gain: self.forward_gain,
            walking_turn_gain: self.turn_gain,
            walking_translation_scale: self.walking_translation_scale,
            brain_wall_seconds,
            brain_encoding_seconds,
            brain_engine_seconds,
        };
        Ok(self.snapshot)
    }

    pub fn reset(&mut self) -> Result<()> {
        let (mut brain, brain_materialization) = load_brain(
            self.pack_path.as_deref(),
            &self.neural_io_path,
            self.parameters.brain,
        )?;
        if let Some(bridge) = brain.as_mut() {
            bridge.set_telemetry_enabled(self.brain_telemetry_enabled);
        }
        self.world.reset()?;
        self.brain = brain;
        self.brain_materialization = brain_materialization;
        self.phase_rad = 0.0;
        self.forward_gain = 1.0;
        self.turn_gain = 0.0;
        self.walking_translation_scale = 1.0;
        self.landing_target_mm = None;
        self.wall_landing = None;
        self.touchdown_gait_ramp = 1.0;
        self.standing_joint_controls =
            std::array::from_fn(|index| self.world.neutral_control()[index]);
        self.feeding_pose_held = false;
        self.feeding_extension = 0.0;
        self.olfactory_transducer.reset();
        self.explorer.reset(0x5eed_f17b_2026_0816);
        self.flight_behavior.reset(0xa17f_1eaf_2026_0816);
        self.foraging.reset();
        self.odor_guidance.reset();
        self.navigation.reset();
        self.ground_navigation.reset();
        self.obstacle_sample = self.world.obstacle_sample(180.0)?;
        self.obstacle_sample_elapsed_seconds = 0.0;
        self.wall_escape = WallEscapeState::default();
        self.grooming.reset();
        self.retina_summaries = [RetinaSummary::default(); 2];
        self.snapshot = SimulationSnapshot {
            root_position: self.world.root_position(),
            food_center: self.food_center,
            food_enabled: self.food_enabled,
            flight_allowed: self.flight_allowed,
            flight_altitude_bounds_mm: self.habitat.room().flight_altitude_bounds_mm,
            forward_gain: 1.0,
            ..SimulationSnapshot::default()
        };
        self.refresh_environment_snapshot()
    }

    pub fn world(&self) -> &MuJoCoWorld {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut MuJoCoWorld {
        &mut self.world
    }

    pub fn snapshot(&self) -> SimulationSnapshot {
        self.snapshot
    }

    pub fn parameters(&self) -> SimulationParameters {
        self.parameters
    }

    pub fn brain_device_name(&self) -> Option<&str> {
        self.brain.as_ref().map(BrainBodyBridge::device_name)
    }

    pub fn brain_allocated_bytes(&self) -> Option<usize> {
        self.brain.as_ref().map(BrainBodyBridge::allocated_bytes)
    }

    pub fn brain_model_name(&self) -> Option<&'static str> {
        self.brain.as_ref().map(BrainBodyBridge::model_name)
    }

    pub fn brain_materialization(&self) -> Option<&str> {
        self.brain_materialization.as_deref()
    }

    pub fn full_neural_io_enabled(&self) -> bool {
        self.brain
            .as_ref()
            .is_some_and(BrainBodyBridge::full_neural_io_enabled)
    }

    pub fn brain_neuron_count(&self) -> usize {
        self.brain.as_ref().map_or(0, BrainBodyBridge::neuron_count)
    }

    pub fn brain_sensory_neuron_count(&self) -> usize {
        self.brain
            .as_ref()
            .map_or(0, BrainBodyBridge::sensory_neuron_count)
    }

    pub fn brain_sensory_neuron_ids(&self) -> &[u64] {
        self.brain
            .as_ref()
            .map_or(&[], BrainBodyBridge::sensory_neuron_ids)
    }

    pub fn brain_motor_neuron_id(&self) -> Option<u64> {
        self.brain.as_ref().map(BrainBodyBridge::motor_neuron_id)
    }

    pub fn neural_io_stats(&self) -> NeuralIoStats {
        self.brain
            .as_ref()
            .map_or_else(NeuralIoStats::default, BrainBodyBridge::neural_io_stats)
    }

    pub fn brain_enabled(&self) -> bool {
        self.brain.is_some()
    }

    pub fn resource_label(&self, resource: Option<usize>) -> &str {
        resource
            .and_then(|index| self.habitat.resources().get(index))
            .map_or("none", |resource| resource.id.as_str())
    }

    pub fn obstacle_label(&self, geom_id: Option<usize>) -> &str {
        self.world.geom_name(geom_id)
    }

    pub fn set_brain_telemetry_enabled(&mut self, enabled: bool) {
        self.brain_telemetry_enabled = enabled;
        if let Some(brain) = self.brain.as_mut() {
            brain.set_telemetry_enabled(enabled);
        }
    }

    pub fn set_retina_summaries(&mut self, summaries: [RetinaSummary; 2]) -> Result<()> {
        if summaries.iter().any(|summary| {
            !summary.mean_intensity.is_finite()
                || !(0.0..=1.0).contains(&summary.mean_intensity)
                || !summary.spatial_contrast.is_finite()
                || summary.spatial_contrast < 0.0
        }) {
            bail!("retina summaries contain invalid values")
        }
        self.retina_summaries = summaries;
        self.snapshot.visual_left = summaries[0].mean_intensity;
        self.snapshot.visual_right = summaries[1].mean_intensity;
        self.snapshot.visual_contrast_left = summaries[0].spatial_contrast;
        self.snapshot.visual_contrast_right = summaries[1].spatial_contrast;
        Ok(())
    }

    pub fn control_period(&self) -> Duration {
        Duration::from_secs_f64(self.control_steps as f64 * self.world.timestep_seconds())
    }

    pub fn food_center(&self) -> [f64; 3] {
        self.food_center
    }

    pub fn set_food_center(&mut self, center: [f64; 3]) -> Result<()> {
        if center.iter().any(|value| !value.is_finite()) {
            bail!("food center must contain finite values")
        }
        self.food_center = center;
        self.refresh_environment_snapshot()
    }

    pub fn move_food(&mut self, delta: [f64; 3]) -> Result<()> {
        self.set_food_center(std::array::from_fn(|axis| {
            self.food_center[axis] + delta[axis]
        }))
    }

    pub fn place_food_at_mouth(&mut self) -> Result<()> {
        let source = self.world.metadata().environment.taste_source_body.clone();
        self.set_food_center(self.world.body_position(&source)?)
    }

    pub fn drop_food_below_fly(&mut self) -> Result<()> {
        let source = self.world.metadata().environment.taste_source_body.clone();
        let mouth = self.world.body_position(&source)?;
        let root = self.world.root_position();
        let support = self.world.obstacle_sample(180.0)?;
        let surface_height = (root[2] - support.down_clearance_mm).max(0.0);
        self.set_food_center([
            mouth[0],
            mouth[1],
            surface_height + self.world.metadata().environment.food_center[2],
        ])
    }

    pub fn place_food_ahead(&mut self, distance: f64) -> Result<()> {
        if !distance.is_finite() || distance <= 0.0 {
            bail!("food distance must be finite and positive")
        }
        let root = self.world.root_position();
        self.set_food_center([
            root[0] + distance,
            root[1],
            self.world.metadata().environment.food_center[2],
        ])
    }

    pub fn set_initial_yaw(&mut self, yaw_rad: f64) -> Result<()> {
        if !yaw_rad.is_finite() || self.world.time() != 0.0 {
            bail!("initial yaw requires a finite angle and an unstepped simulation")
        }
        let [w, x, y, z] = self.world.qpos()[3..7].try_into().unwrap();
        let (s, c) = (0.5 * yaw_rad).sin_cos();
        self.world.data_mut().qpos_mut()[3..7].copy_from_slice(&[
            c * w - s * z,
            c * x - s * y,
            c * y + s * x,
            c * z + s * w,
        ]);
        self.world.data_mut().forward();
        self.refresh_environment_snapshot()
    }

    pub fn toggle_food(&mut self) -> Result<()> {
        self.food_enabled = !self.food_enabled;
        self.refresh_environment_snapshot()
    }

    pub fn toggle_flight(&mut self) {
        self.flight_allowed = !self.flight_allowed;
        self.snapshot.flight_allowed = self.flight_allowed;
    }

    pub fn flight_allowed(&self) -> bool {
        self.flight_allowed
    }

    pub fn request_grooming(&mut self) {
        self.grooming.request_manual();
    }

    fn refresh_environment_snapshot(&mut self) -> Result<()> {
        let WorldEnvironment {
            taste_source_body, ..
        } = self.world.metadata().environment.clone();
        let taste_position = self.world.body_position(&taste_source_body)?;
        let left_antenna = self.world.body_position("fly/l_funiculus")?;
        let right_antenna = self.world.body_position("fly/r_funiculus")?;
        let habitat_sample = self.habitat.sample(
            left_antenna,
            right_antenna,
            taste_position,
            self.food_center,
            self.food_enabled,
        );
        let olfactory_sample = self
            .olfactory_transducer
            .preview([habitat_sample.odor_left_ppm, habitat_sample.odor_right_ppm])?;
        let food_distance = distance(taste_position, self.food_center);
        self.snapshot.time_seconds = self.world.time();
        self.snapshot.root_position = self.world.root_position();
        let root_velocity = self.world.root_velocity();
        self.snapshot.horizontal_speed_mm_s = root_velocity[3].hypot(root_velocity[4]);
        self.snapshot.forward_speed_mm_s = planar_forward_speed(&self.world);
        self.snapshot.body_pitch_deg = body_pitch_deg(&self.world);
        self.snapshot.food_center = self.food_center;
        self.snapshot.food_distance = food_distance;
        self.snapshot.food_enabled = self.food_enabled;
        self.snapshot.odor_left = olfactory_sample.perceived_intensity[0];
        self.snapshot.odor_right = olfactory_sample.perceived_intensity[1];
        self.snapshot.odor_left_ppm = olfactory_sample.concentration_ppm[0];
        self.snapshot.odor_right_ppm = olfactory_sample.concentration_ppm[1];
        self.snapshot.odor_intensity = 0.5
            * (olfactory_sample.perceived_intensity[0] + olfactory_sample.perceived_intensity[1]);
        self.snapshot.taste_active = habitat_sample.tasted_resource.is_some();
        self.snapshot.tasted_resource = habitat_sample.tasted_resource;
        self.snapshot.nearest_resource = habitat_sample.nearest_resource;
        self.snapshot.nearest_resource_distance = habitat_sample.nearest_distance_mm;
        Ok(())
    }
}

fn gait_excursion_gain(activation: f64, feeding_extension: f64, cns: bool) -> f64 {
    if cns {
        f64::from(activation > 0.0 && feeding_extension <= 0.1)
    } else {
        activation
    }
}

fn walking_side_drive(excursion: f64, translation_scale: f64, turn: f64) -> [f64; 2] {
    [
        excursion * (translation_scale - 0.5 * turn),
        excursion * (translation_scale + 0.5 * turn),
    ]
}

fn load_brain(
    pack_path: Option<&Path>,
    neural_io_path: &Path,
    parameters: BrainBridgeParameters,
) -> Result<(Option<BrainBodyBridge>, Option<String>)> {
    let Some(pack_path) = pack_path else {
        return Ok((None, None));
    };
    let pack = ConnectomePack::open(pack_path)?;
    let materialization = pack.materialization().to_string();
    let brain = if pack.materialization() == MALE_CNS_MATERIALIZATION {
        BrainBodyBridge::new_with_neural_io_and_parameters(
            &pack,
            neural_io_path.with_file_name(MALE_CNS_NEURAL_IO_FILE),
            parameters,
        )?
    } else if pack.materialization() == "783" {
        BrainBodyBridge::new_with_neural_io_and_parameters(&pack, neural_io_path, parameters)?
    } else {
        BrainBodyBridge::new_with_parameters(&pack, parameters)?
    };
    Ok((Some(brain), Some(materialization)))
}

fn rounded_positive_ratio(numerator: f64, denominator: f64) -> Result<usize> {
    let ratio = numerator / denominator;
    if !ratio.is_finite() || ratio < 1.0 {
        bail!("control period must contain at least one physics step")
    }
    let rounded = ratio.round();
    if (ratio - rounded).abs() > 1e-8 * ratio.max(1.0) {
        bail!("control period must be an integer multiple of the physics timestep")
    }
    Ok(rounded as usize)
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn clamp_joint_controls_to_actuator_ranges(
    world: &MuJoCoWorld,
    controls: &mut [f64; 42],
) -> Result<()> {
    for (index, control) in controls.iter_mut().enumerate() {
        let [minimum, maximum] = world.actuator_control_range(index)?;
        *control = control.clamp(minimum, maximum);
    }
    Ok(())
}

fn flight_frequency_scale(
    mode: FlightMode,
    height_error_mm: f64,
    vertical_velocity_mm_s: f64,
) -> f64 {
    match mode {
        FlightMode::Grounded => 1.0,
        FlightMode::Takeoff => {
            (1.32 + 0.004 * height_error_mm - 0.0005 * vertical_velocity_mm_s).clamp(1.25, 1.5)
        }
        FlightMode::Cruise | FlightMode::Landing => {
            (1.25 + 0.004 * height_error_mm - 0.0005 * vertical_velocity_mm_s).clamp(1.05, 1.45)
        }
    }
}

fn commanded_flight_height(
    mode: FlightMode,
    desired_height_mm: f64,
    current_height_mm: f64,
    minimum_height_mm: f64,
    overhead: bool,
    altitude_escape: bool,
) -> f64 {
    if matches!(mode, FlightMode::Grounded | FlightMode::Landing) {
        return desired_height_mm;
    }
    if altitude_escape {
        return desired_height_mm
            .min((current_height_mm - OVERHANG_DESCENT_MM).max(minimum_height_mm))
            .max(minimum_height_mm);
    }
    if overhead {
        desired_height_mm
            .min(current_height_mm)
            .max(minimum_height_mm)
    } else {
        desired_height_mm
    }
}

fn nearby_wall_landing_target(
    world: &MuJoCoWorld,
    maximum_distance_mm: f64,
) -> Option<WallLandingTarget> {
    let position = world.root_position();
    let model = world.model();
    (0..model.ngeom() as usize)
        .filter_map(|geom| {
            let name = model.id_to_name(MjtObj::mjOBJ_GEOM, geom)?;
            if !name.starts_with("room_wall_") || name == "room_wall_ceiling" {
                return None;
            }
            let center = world.data().geom_xpos()[geom];
            let size = model.geom_size()[geom];
            let axis = usize::from(size[1] < size[0]);
            let tangent = 1 - axis;
            if (position[tangent] - center[tangent]).abs() > size[tangent]
                || position[2] < center[2] - size[2] + 4.0
                || position[2] > center[2] + size[2] - 4.0
            {
                return None;
            }
            let inward = -center[axis].signum();
            let surface = center[axis] + inward * size[axis];
            let distance = (position[axis] - surface) * inward;
            if !(0.0..=maximum_distance_mm).contains(&distance) {
                return None;
            }
            let mut surface_point_mm = position;
            surface_point_mm[axis] = surface;
            let mut inward_xy = [0.0; 2];
            inward_xy[axis] = inward;
            Some((
                distance,
                WallLandingTarget {
                    surface_point_mm,
                    inward_xy,
                },
            ))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, target)| target)
}

fn surface_relative_landing_height(
    current_height_mm: f64,
    down_clearance_mm: f64,
    landing_clearance_mm: f64,
) -> f64 {
    (current_height_mm - down_clearance_mm + landing_clearance_mm).max(landing_clearance_mm)
}

fn grounded_navigation_forward_gain(
    forward_clearance_mm: f64,
    collision_reflex_active: bool,
) -> f64 {
    if collision_reflex_active {
        return GROUNDED_COLLISION_FORWARD_GAIN;
    }
    if forward_clearance_mm > GROUNDED_OBSTACLE_SLOWING_DISTANCE_MM {
        return 1.0;
    }
    let clearance_scale =
        (forward_clearance_mm / GROUNDED_OBSTACLE_SLOWING_DISTANCE_MM).clamp(0.0, 1.0);
    GROUNDED_COLLISION_FORWARD_GAIN
        + clearance_scale * (GROUNDED_OBSTACLE_FORWARD_GAIN - GROUNDED_COLLISION_FORWARD_GAIN)
}

fn planar_forward(root_quaternion: [f64; 4]) -> [f64; 2] {
    let [w, x, y, z] = root_quaternion;
    let forward = [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z)];
    let norm = forward[0].hypot(forward[1]).max(1e-9);
    [forward[0] / norm, forward[1] / norm]
}

fn dot_xy(left: [f64; 2], right: [f64; 2]) -> f64 {
    left[0] * right[0] + left[1] * right[1]
}

fn wall_escape_speed_scale(forward_xy: [f64; 2], escape_direction_xy: [f64; 2]) -> f64 {
    ((dot_xy(forward_xy, escape_direction_xy) - WALL_ESCAPE_ACCELERATION_ALIGNMENT)
        / (1.0 - WALL_ESCAPE_ACCELERATION_ALIGNMENT))
        .clamp(0.0, 1.0)
}

fn wall_inward_direction(position_mm: [f64; 3], room_half_extents_mm: [f64; 3]) -> [f64; 2] {
    let x_clearance = room_half_extents_mm[0] - position_mm[0].abs();
    let y_clearance = room_half_extents_mm[1] - position_mm[1].abs();
    if x_clearance <= PLANAR_WALL_ESCAPE_RELEASE_MM && y_clearance <= PLANAR_WALL_ESCAPE_RELEASE_MM
    {
        let scale = std::f64::consts::FRAC_1_SQRT_2;
        return [
            -position_mm[0].signum() * scale,
            -position_mm[1].signum() * scale,
        ];
    }
    [
        (x_clearance, [-position_mm[0].signum(), 0.0]),
        (y_clearance, [0.0, -position_mm[1].signum()]),
    ]
    .into_iter()
    .min_by(|left, right| left.0.total_cmp(&right.0))
    .expect("a room has four planar walls")
    .1
}

fn planar_wall_clearance(position_mm: [f64; 3], room_half_extents_mm: [f64; 3]) -> f64 {
    (room_half_extents_mm[0] - position_mm[0].abs())
        .min(room_half_extents_mm[1] - position_mm[1].abs())
}

fn full_brain_walking_turn(
    mode: BehaviorMode,
    procedural_turn: f64,
    brain_steering: f64,
    brain_steering_gain: f64,
) -> f64 {
    match mode {
        BehaviorMode::Explore | BehaviorMode::TrackOdor => {
            (brain_steering_gain * brain_steering).clamp(-1.0, 1.0)
        }
        BehaviorMode::Feed => 0.0,
        BehaviorMode::DepartFood => procedural_turn,
    }
}

fn full_brain_walking_forward(
    mode: BehaviorMode,
    procedural_forward: f64,
    brain_forward: f64,
) -> f64 {
    match mode {
        BehaviorMode::Explore | BehaviorMode::TrackOdor => brain_forward.clamp(0.0, 1.0),
        BehaviorMode::Feed => 0.0,
        BehaviorMode::DepartFood => procedural_forward,
    }
}

fn grounded_food_contact_blocks_flight(
    previous_flight_mode: FlightMode,
    taste_active: bool,
    behavior_mode: BehaviorMode,
) -> bool {
    previous_flight_mode == FlightMode::Grounded
        && taste_active
        && matches!(behavior_mode, BehaviorMode::Feed | BehaviorMode::DepartFood)
}

fn optic_flow_altitude_control(
    horizontal_speed_mm_s: f64,
    root_height_mm: f64,
    down_clearance_mm: f64,
    overhead: bool,
) -> f64 {
    if overhead || horizontal_speed_mm_s < 50.0 {
        return 0.0;
    }
    let optic_flow_rad_s = horizontal_speed_mm_s / down_clearance_mm.max(3.0);
    let flow_ratio = optic_flow_rad_s / 4.0;
    if flow_ratio > 1.15 {
        ((flow_ratio - 1.15) / 1.35).clamp(0.0, 1.0)
    } else if flow_ratio < 0.65 && root_height_mm > 40.0 {
        -((0.65 - flow_ratio) / 0.65).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn planar_forward_speed(world: &MuJoCoWorld) -> f64 {
    let forward = planar_forward(world.root_quaternion());
    let velocity = world.root_velocity();
    velocity[3] * forward[0] + velocity[4] * forward[1]
}

fn body_pitch_deg(world: &MuJoCoWorld) -> f64 {
    let [w, x, y, z] = world.root_quaternion();
    (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin().to_degrees()
}

#[cfg(test)]
mod tests {
    #[test]
    fn cns_activation_controls_cadence_without_shrinking_stride_geometry() {
        assert_eq!(super::gait_excursion_gain(0.5, 0.0, true), 1.0);
        assert_eq!(super::gait_excursion_gain(0.0, 0.0, true), 0.0);
        assert_eq!(super::gait_excursion_gain(0.01, 0.99, true), 0.0);
        assert_eq!(super::gait_excursion_gain(0.4, 0.35, true), 0.0);
        assert_eq!(super::gait_excursion_gain(0.0, 1.0, true), 0.0);
        assert_eq!(super::gait_excursion_gain(0.5, 0.5, false), 0.5);
    }

    #[test]
    fn obstacle_braking_preserves_turning_stride_and_neural_gating() {
        let straight = super::walking_side_drive(1.0, 1.0, 0.7);
        let braking = super::walking_side_drive(1.0, 0.08, 0.7);
        assert!(braking[0] < 0.0 && braking[1] > 0.0);
        assert!(((straight[1] - straight[0]) - (braking[1] - braking[0])).abs() < 1e-12);
        assert_eq!(super::walking_side_drive(0.0, 0.08, 0.7), [0.0, 0.0]);
        let feeding = super::walking_side_drive(0.1, 0.08, 0.7);
        assert!((feeding[1] - 0.1 * braking[1]).abs() < 1e-12);
    }

    #[test]
    fn initial_yaw_changes_heading_and_rejects_midrun_teleportation() {
        let mut simulation = super::SimulationStepper::new(
            crate::world::DEFAULT_ASSETS_DIR,
            None::<&str>,
            500.0,
            0.5,
        )
        .unwrap();
        simulation
            .set_initial_yaw(std::f64::consts::FRAC_PI_2)
            .unwrap();
        let heading = super::planar_forward(simulation.world().root_quaternion());
        assert!(heading[0].abs() < 1e-9 && (heading[1] - 1.0).abs() < 1e-9);
        assert_eq!(
            simulation.snapshot().root_position,
            simulation.world().root_position()
        );
        simulation.step_window().unwrap();
        assert!(simulation.set_initial_yaw(0.0).is_err());
    }

    use super::{
        PLANAR_WALL_ESCAPE_RELEASE_MM, SimulationParameters, SimulationStepper,
        WALL_ESCAPE_RELEASE_ALIGNMENT, WALL_ESCAPE_RELEASE_DWELL_WINDOWS,
        WALL_ESCAPE_RELEASE_INWARD_SPEED_MM_S, WallEscapeObservation, WallEscapeState,
        commanded_flight_height, dot_xy, flight_frequency_scale, full_brain_walking_forward,
        full_brain_walking_turn, grounded_food_contact_blocks_flight,
        grounded_navigation_forward_gain, optic_flow_altitude_control, planar_forward,
        planar_wall_clearance, surface_relative_landing_height, wall_escape_speed_scale,
        wall_inward_direction,
    };
    use crate::behavior::BehaviorMode;
    use crate::flight::WallLandingTarget;
    use crate::flight_behavior::{FlightBehaviorInput, FlightMode, TAKEOFF_DRIVE_THRESHOLD};
    use crate::grooming::GroomingTrigger;
    use crate::world::{DEFAULT_ASSETS_DIR, MuJoCoWorld};
    use mujoco_rs::prelude::*;

    fn prime_airborne_simulation(
        simulation: &mut SimulationStepper,
        position_mm: [f64; 3],
        planar_velocity_mm_s: [f64; 3],
    ) {
        let mut command = Default::default();
        for _ in 0..100 {
            command = simulation
                .flight_behavior
                .update(FlightBehaviorInput {
                    dt_seconds: 0.002,
                    enabled: true,
                    root_height_mm: 1.0,
                    brain_flight_drive: TAKEOFF_DRIVE_THRESHOLD + 0.02,
                    ..FlightBehaviorInput::default()
                })
                .unwrap();
        }
        assert_eq!(command.mode, FlightMode::Takeoff);
        simulation.snapshot.flight_mode = FlightMode::Takeoff;
        {
            let half_pitch = 0.5 * simulation.parameters.flight_dynamics.target_pitch_rad;
            let data = simulation.world_mut().data_mut();
            data.qpos_mut()[..3].copy_from_slice(&position_mm);
            data.qpos_mut()[3..7].copy_from_slice(&[half_pitch.cos(), 0.0, half_pitch.sin(), 0.0]);
            data.qvel_mut()[..6].fill(0.0);
            data.qvel_mut()[..3].copy_from_slice(&planar_velocity_mm_s);
            data.forward();
        }
        simulation.refresh_environment_snapshot().unwrap();
    }

    #[test]
    fn baseline_parameter_artifact_matches_compiled_defaults() {
        let loaded = SimulationParameters::load(format!(
            "{DEFAULT_ASSETS_DIR}/simulation_parameters_baseline_v1.json"
        ))
        .unwrap();
        assert_eq!(loaded, SimulationParameters::default());
    }

    #[test]
    fn food_target_can_move_without_advancing_physics() {
        let mut simulation =
            SimulationStepper::new(DEFAULT_ASSETS_DIR, None::<&str>, 500.0, 0.5).unwrap();
        let initial_time = simulation.world().time();
        simulation.place_food_ahead(3.0).unwrap();
        assert_eq!(simulation.world().time(), initial_time);
        assert!(!simulation.snapshot().taste_active);
        simulation.place_food_at_mouth().unwrap();
        assert!(simulation.snapshot().taste_active);
        simulation.toggle_food().unwrap();
        assert!(!simulation.snapshot().taste_active);
    }

    #[test]
    fn landing_height_tracks_the_surface_below_the_fly() {
        assert_eq!(surface_relative_landing_height(28.0, 28.0, 1.4), 1.4);
        assert_eq!(surface_relative_landing_height(72.0, 20.0, 1.4), 53.4);
        assert_eq!(surface_relative_landing_height(1.2, 1.8, 1.4), 1.4);
    }

    #[test]
    fn grounded_obstacle_reflex_slows_before_contact() {
        let approaching = grounded_navigation_forward_gain(55.0, false);
        let near = grounded_navigation_forward_gain(8.0, false);
        let collision = grounded_navigation_forward_gain(6.0, true);
        assert!(approaching > near);
        assert!(near > collision);
        assert_eq!(collision, 0.08);
    }

    #[test]
    fn completed_meal_releases_the_fly_from_the_food_contact() {
        let mut simulation =
            SimulationStepper::new(DEFAULT_ASSETS_DIR, None::<&str>, 500.0, 0.5).unwrap();
        for _ in 0..300 {
            simulation.step_window().unwrap();
        }
        simulation.place_food_at_mouth().unwrap();
        let initial_position = simulation.world().root_position();
        let mut saw_feeding = false;
        let mut saw_departure = false;
        let mut final_snapshot = simulation.snapshot();
        for _ in 0..6_000 {
            final_snapshot = simulation.step_window().unwrap();
            saw_feeding |= final_snapshot.behavior_mode == BehaviorMode::Feed;
            saw_departure |= final_snapshot.behavior_mode == BehaviorMode::DepartFood;
            if saw_departure && !final_snapshot.taste_active {
                break;
            }
        }
        let displacement = (final_snapshot.root_position[0] - initial_position[0])
            .hypot(final_snapshot.root_position[1] - initial_position[1]);
        assert!(saw_feeding);
        assert!(saw_departure);
        assert!(!final_snapshot.taste_active);
        assert!(displacement > 0.25);
        assert!(final_snapshot.forward_gain > 0.0);
    }

    #[test]
    fn post_meal_locomotion_stays_upright() {
        let mut simulation =
            SimulationStepper::new(DEFAULT_ASSETS_DIR, None::<&str>, 500.0, 0.5).unwrap();
        for _ in 0..300 {
            simulation.step_window().unwrap();
        }
        simulation.place_food_at_mouth().unwrap();

        let mut saw_departure = false;
        let mut minimum_up_z = 1.0_f64;
        let mut posture_failure = None;
        for step in 0..10_000 {
            let snapshot = simulation.step_window().unwrap();
            saw_departure |= snapshot.behavior_mode == BehaviorMode::DepartFood;
            let quaternion = simulation.world().root_quaternion();
            let [_, x, y, _] = quaternion;
            let up_z = 1.0 - 2.0 * (x * x + y * y);
            minimum_up_z = minimum_up_z.min(up_z);
            if up_z <= 0.5 {
                posture_failure = Some((step, snapshot, quaternion));
                break;
            }
        }

        assert!(saw_departure);
        assert!(
            posture_failure.is_none(),
            "the thorax lost its upright posture after feeding: minimum body up-z={minimum_up_z}, failure={posture_failure:?}"
        );
    }

    #[test]
    fn airborne_taste_keeps_wings_and_stabilization_active() {
        let mut simulation =
            SimulationStepper::new(DEFAULT_ASSETS_DIR, None::<&str>, 500.0, 0.0).unwrap();
        prime_airborne_simulation(&mut simulation, [0.0, 0.0, 30.0], [120.0, 0.0, 0.0]);
        simulation.place_food_at_mouth().unwrap();

        let mut minimum_height = f64::INFINITY;
        let mut maximum_abs_pitch = 0.0_f64;
        for _ in 0..250 {
            let snapshot = simulation.step_window().unwrap();
            assert_ne!(snapshot.flight_mode, FlightMode::Grounded);
            minimum_height = minimum_height.min(snapshot.root_position[2]);
            maximum_abs_pitch = maximum_abs_pitch.max(snapshot.body_pitch_deg.abs());
        }
        assert!(minimum_height > 5.0);
        assert!(maximum_abs_pitch < 89.0);
    }

    #[test]
    fn airborne_taste_does_not_start_the_meal_clock() {
        let mut parameters = SimulationParameters::default();
        parameters.behavior.feeding_bout_seconds = 0.1;
        let mut simulation = SimulationStepper::new_with_parameters(
            DEFAULT_ASSETS_DIR,
            None::<&str>,
            500.0,
            0.0,
            parameters,
        )
        .unwrap();
        prime_airborne_simulation(&mut simulation, [0.0, 0.0, 30.0], [120.0, 0.0, 0.0]);
        for _ in 0..200 {
            simulation.place_food_at_mouth().unwrap();
            let snapshot = simulation.step_window().unwrap();
            assert!(snapshot.taste_active);
            assert_ne!(snapshot.flight_mode, FlightMode::Grounded);
            assert!(!matches!(
                snapshot.behavior_mode,
                BehaviorMode::Feed | BehaviorMode::DepartFood
            ));
        }
    }

    #[test]
    fn airborne_walking_phase_is_paused() {
        let mut simulation =
            SimulationStepper::new(DEFAULT_ASSETS_DIR, None::<&str>, 500.0, 0.0).unwrap();
        prime_airborne_simulation(&mut simulation, [0.0, 0.0, 30.0], [120.0, 0.0, 0.0]);
        simulation.phase_rad = 1.234;
        for _ in 0..20 {
            let snapshot = simulation.step_window().unwrap();
            assert_ne!(snapshot.flight_mode, FlightMode::Grounded);
            assert_eq!(simulation.phase_rad, 1.234);
        }
    }

    #[test]
    fn landing_on_food_levels_brakes_and_descends_before_support_handoff() {
        let mut simulation =
            SimulationStepper::new(DEFAULT_ASSETS_DIR, None::<&str>, 500.0, 0.0).unwrap();
        prime_airborne_simulation(&mut simulation, [29.8, 14.6, 8.0], [4.0, 4.0, 0.0]);
        let half_yaw = std::f64::consts::FRAC_PI_4 * 0.5;
        let half_pitch = simulation.parameters.flight_dynamics.target_pitch_rad * 0.5;
        simulation.world.data_mut().qpos_mut()[3..7].copy_from_slice(&[
            half_yaw.cos() * half_pitch.cos(),
            -half_yaw.sin() * half_pitch.sin(),
            half_yaw.cos() * half_pitch.sin(),
            half_yaw.sin() * half_pitch.cos(),
        ]);
        simulation.world.data_mut().forward();
        let command = simulation
            .flight_behavior
            .update(FlightBehaviorInput {
                enabled: true,
                landing_request: true,
                dt_seconds: 0.002,
                root_height_mm: 8.0,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(command.mode, FlightMode::Landing);
        let mut touchdown = None;
        let mut tasted = false;
        let mut minimum_up_z = 1.0_f64;
        let mut peak_ground_speed = 0.0_f64;
        for _ in 0..1000 {
            let before_velocity = simulation.world.root_velocity();
            let snapshot = simulation.step_window().unwrap();
            if snapshot.flight_mode == FlightMode::Landing {
                assert_eq!(snapshot.flight_horizontal_speed_scale, 0.0);
            }
            if snapshot.flight_mode == FlightMode::Grounded && touchdown.is_none() {
                let [_, x, y, _] = simulation.world.root_quaternion();
                touchdown = Some((
                    snapshot.time_seconds,
                    snapshot.root_position,
                    before_velocity[5],
                    1.0 - 2.0 * (x * x + y * y),
                ));
            }
            tasted |= snapshot.taste_active;
            if touchdown.is_some() {
                let [_, x, y, _] = simulation.world.root_quaternion();
                minimum_up_z = minimum_up_z.min(1.0 - 2.0 * (x * x + y * y));
                peak_ground_speed = peak_ground_speed.max(snapshot.horizontal_speed_mm_s);
            }
        }
        let (time, position, vz, up_z) = touchdown.unwrap_or_else(|| {
            panic!(
                "food touchdown missing: {:?}, contacts {:?}, velocity {:?}, amplitude {}",
                simulation.snapshot.root_position,
                simulation.world.support_contacts(),
                simulation.world.root_velocity(),
                simulation.snapshot.flight_amplitude_scale
            )
        });
        eprintln!(
            "food touchdown t={time} position={position:?} vz={vz} up_z={up_z} minimum_up_z={minimum_up_z} peak_ground_speed={peak_ground_speed} tasted={tasted} final={:?} quat={:?} contacts={:?} vel={:?}",
            simulation.snapshot.root_position,
            simulation.world.root_quaternion(),
            simulation.world.support_contacts().unwrap(),
            simulation.world.root_velocity()
        );
        assert!(time > 0.2 && time < 1.5, "touchdown t={time}");
        assert!(vz.abs() < 30.0, "touchdown vz={vz}");
        assert!(up_z > 0.9, "touchdown up_z={up_z} position={position:?}");
        assert!(
            (position[0] - 29.8).hypot(position[1] - 14.6) < 5.0,
            "touchdown={position:?}"
        );
        let [_, x, y, _] = simulation.world.root_quaternion();
        assert!(1.0 - 2.0 * (x * x + y * y) > 0.8);
        assert!(minimum_up_z > 0.7, "minimum_up_z={minimum_up_z}");
        assert!(
            peak_ground_speed < 50.0,
            "peak_ground_speed={peak_ground_speed}"
        );
        assert!(tasted);
        assert_eq!(
            &simulation.world.controls()[crate::world::WING_ACTUATOR_START..],
            &[1.5, -1.0, 0.7, 1.5, -1.0, 0.7]
        );
        assert!(
            simulation
                .world
                .wing_joint_velocities()
                .iter()
                .all(|speed| speed.abs() < 2.0)
        );
    }

    fn minimum_wall_mesh_clearance(world: &MuJoCoWorld, target: WallLandingTarget) -> f64 {
        let model = world.model();
        let data = world.data();
        let mut minimum = f64::INFINITY;
        for geom in 0..model.ngeom() as usize {
            let name = world.geom_name(Some(geom));
            if !name.starts_with("fly/") || model.geom_contype()[geom] == 0 {
                continue;
            }
            let mesh = model.geom_dataid()[geom];
            if model.geom_type()[geom] != MjtGeom::mjGEOM_MESH || mesh < 0 {
                continue;
            }
            let mesh = mesh as usize;
            let start = model.mesh_vertadr()[mesh] as usize;
            let count = model.mesh_vertnum()[mesh] as usize;
            let position = data.geom_xpos()[geom];
            let rotation = data.geom_xmat()[geom];
            for vertex in &model.mesh_vert()[start..start + count] {
                let distance = (0..2)
                    .map(|axis| {
                        let coordinate = position[axis]
                            + (0..3)
                                .map(|k| rotation[axis * 3 + k] * f64::from(vertex[k]))
                                .sum::<f64>();
                        (coordinate - target.surface_point_mm[axis]) * target.inward_xy[axis]
                    })
                    .sum::<f64>();
                minimum = minimum.min(distance);
            }
        }
        minimum
    }

    #[test]
    fn wall_landing_aligns_attaches_and_parks_wings() {
        for (position, yaw, pitch) in [
            ([290.0, 0.0, 100.0], 0.0, -0.83),
            ([-290.0, 0.0, 100.0], std::f64::consts::PI, -0.83),
            ([0.0, 210.0, 100.0], std::f64::consts::FRAC_PI_2, -0.83),
            ([0.0, -210.0, 100.0], -std::f64::consts::FRAC_PI_2, -0.83),
            ([296.0, 0.0, 132.0], 0.0, 0.0),
            ([-296.0, 0.0, 132.0], std::f64::consts::FRAC_PI_2, 0.0),
        ] {
            let mut simulation =
                SimulationStepper::new(DEFAULT_ASSETS_DIR, None::<&str>, 500.0, 0.0).unwrap();
            prime_airborne_simulation(&mut simulation, position, [0.0; 3]);
            let half_pitch: f64 = pitch * 0.5;
            let half_yaw = yaw * 0.5;
            simulation.world.data_mut().qpos_mut()[3..7].copy_from_slice(&[
                half_yaw.cos() * half_pitch.cos(),
                -half_yaw.sin() * half_pitch.sin(),
                half_yaw.cos() * half_pitch.sin(),
                half_yaw.sin() * half_pitch.cos(),
            ]);
            simulation.world.data_mut().forward();
            for _ in 0..2 {
                simulation
                    .flight_behavior
                    .update(FlightBehaviorInput {
                        enabled: true,
                        landing_request: true,
                        dt_seconds: 0.002,
                        root_height_mm: position[2],
                        ..Default::default()
                    })
                    .unwrap();
            }
            let mut touchdown = None;
            let mut minimum_height = position[2];
            let mut minimum_mesh_clearance = f64::INFINITY;
            for _ in 0..2000 {
                let snapshot = simulation.step_window().unwrap();
                minimum_mesh_clearance = minimum_mesh_clearance.min(minimum_wall_mesh_clearance(
                    &simulation.world,
                    simulation.wall_landing.unwrap(),
                ));
                minimum_height = minimum_height.min(snapshot.root_position[2]);
                if snapshot.flight_mode == FlightMode::Grounded && touchdown.is_none() {
                    touchdown = Some(snapshot.time_seconds);
                }
                if let Some(time) = touchdown {
                    assert_eq!(
                        snapshot.flight_mode,
                        FlightMode::Grounded,
                        "wall support relapsed at {}",
                        snapshot.time_seconds
                    );
                    assert!(
                        snapshot.wall_support_leg_count >= 2,
                        "lost wall support at {}",
                        snapshot.time_seconds
                    );
                    assert_eq!(snapshot.flight_amplitude_scale, 0.0);
                    assert!(
                        minimum_mesh_clearance >= -0.01,
                        "wall mesh penetration {} mm at {} s, root {:?}",
                        -minimum_mesh_clearance,
                        snapshot.time_seconds,
                        snapshot.root_position
                    );
                    if snapshot.time_seconds > time + 0.5 {
                        break;
                    }
                }
            }
            assert!(
                touchdown.is_some(),
                "wall touchdown missing: position {position:?}, final {:?}",
                simulation.snapshot
            );
            assert!(
                minimum_height > position[2] - 10.0,
                "wall landing dropped to {minimum_height}"
            );
            assert!(
                simulation
                    .wall_landing
                    .unwrap()
                    .alignment(simulation.world.root_quaternion())
                    > 0.95
            );
            assert!(
                simulation
                    .world
                    .wing_joint_velocities()
                    .iter()
                    .all(|v| v.abs() < 2.0)
            );
            assert!(
                simulation
                    .world
                    .data()
                    .warning()
                    .iter()
                    .all(|warning| warning.number == 0)
            );
            eprintln!(
                "wall {position:?}, yaw {yaw}, pitch {pitch}: touchdown {touchdown:?}, final {:?}, contacts {}, minimum mesh clearance {minimum_mesh_clearance} mm, alignment {}",
                simulation.snapshot.root_position,
                simulation.snapshot.wall_support_leg_count,
                simulation
                    .wall_landing
                    .unwrap()
                    .alignment(simulation.world.root_quaternion())
            );
            let inward = simulation.wall_landing.unwrap().inward_xy;
            for _ in 0..100 {
                simulation
                    .flight_behavior
                    .update(FlightBehaviorInput {
                        enabled: true,
                        brain_flight_drive: 1.0,
                        dt_seconds: 0.002,
                        root_height_mm: position[2],
                        ..Default::default()
                    })
                    .unwrap();
            }
            let mut departed = false;
            for _ in 0..2000 {
                let snapshot = simulation.step_window().unwrap();
                assert!(
                    simulation.world.controls()
                        [crate::world::JOINT_ACTUATOR_COUNT..crate::world::WING_ACTUATOR_START]
                        .iter()
                        .all(|&control| control == 0.0)
                );
                if planar_wall_clearance(
                    snapshot.root_position,
                    simulation.habitat.room().half_extents_mm,
                ) > 15.0
                    && dot_xy(planar_forward(simulation.world.root_quaternion()), inward) > 0.7
                {
                    departed = true;
                    break;
                }
            }
            assert!(
                departed,
                "failed to take off from wall {position:?}: {:?}",
                simulation.snapshot
            );
        }
    }

    #[test]
    fn airborne_wall_escape_turns_then_leaves_head_first_without_relatching() {
        let mut parameters = SimulationParameters::default();
        parameters.flight_behavior.landing_odor_threshold = 2.1;
        let mut simulation = SimulationStepper::new_with_parameters(
            DEFAULT_ASSETS_DIR,
            None::<&str>,
            500.0,
            0.0,
            parameters,
        )
        .unwrap();
        simulation.toggle_food().unwrap();
        prime_airborne_simulation(&mut simulation, [0.0, 0.0, 30.0], [250.0, 0.0, 0.0]);

        let mut saw_turn_hold = false;
        let mut escape_windows = 0;
        let mut minimum_escape_height = f64::INFINITY;
        let mut inward = [-1.0, 0.0];
        let mut released_position = None;
        for _ in 0..3_000 {
            let was_latched = simulation.wall_escape.latched;
            let before_forward = planar_forward(simulation.world().root_quaternion());
            let before_velocity = simulation.world().root_velocity();
            let before_position = simulation.world().root_position();
            let snapshot = simulation.step_window().unwrap();
            let forward = planar_forward(simulation.world().root_quaternion());
            let escape_active =
                simulation.wall_escape.latched || simulation.wall_escape.release_hold_windows > 0;
            if escape_active {
                escape_windows += 1;
                minimum_escape_height = minimum_escape_height.min(snapshot.root_position[2]);
                inward = simulation.wall_escape.direction_xy;
                if dot_xy(forward, inward) < 0.0 {
                    saw_turn_hold |= snapshot.flight_horizontal_speed_scale == 0.0;
                }
            }
            if was_latched
                && !simulation.wall_escape.latched
                && simulation.wall_escape.release_hold_windows > 0
            {
                assert!(
                    dot_xy(before_forward, inward) >= WALL_ESCAPE_RELEASE_ALIGNMENT,
                    "wall escape released without head-first alignment: forward={before_forward:?}, inward={inward:?}, alignment={}",
                    dot_xy(before_forward, inward)
                );
                assert!(
                    dot_xy([before_velocity[3], before_velocity[4]], inward)
                        >= WALL_ESCAPE_RELEASE_INWARD_SPEED_MM_S,
                    "wall escape released without inward speed: velocity={:?}, inward={inward:?}, inward_speed={}",
                    [before_velocity[3], before_velocity[4]],
                    dot_xy([before_velocity[3], before_velocity[4]], inward)
                );
                released_position = Some(before_position);
                break;
            }
        }
        let released_position = released_position.unwrap_or_else(|| {
            panic!(
                "wall escape never completed: pos={:?} forward={:?} velocity={:?} mode={} latched={} speed_scale={}",
                simulation.snapshot().root_position,
                planar_forward(simulation.world().root_quaternion()),
                simulation.world().root_velocity(),
                simulation.snapshot().flight_mode.label(),
                simulation.wall_escape.latched,
                simulation.snapshot().flight_horizontal_speed_scale,
            )
        });
        assert!(saw_turn_hold);
        let control_period = simulation.control_steps as f64 * simulation.world.timestep_seconds();
        let half_turn_seconds =
            std::f64::consts::PI / parameters.flight_dynamics.maximum_yaw_rate_rad_s;
        let release_dwell_seconds = WALL_ESCAPE_RELEASE_DWELL_WINDOWS as f64 * control_period;
        assert!(
            escape_windows as f64 * control_period
                < half_turn_seconds + release_dwell_seconds + 0.2,
            "escape_windows={escape_windows}"
        );
        assert!(minimum_escape_height > 20.0);
        assert!(300.0 - released_position[0].abs() >= PLANAR_WALL_ESCAPE_RELEASE_MM);

        let departure_start = released_position;
        for _ in 0..100 {
            simulation.step_window().unwrap();
            assert!(!simulation.wall_escape.latched);
        }
        let departure = simulation.snapshot().root_position;
        assert!(
            dot_xy(
                [
                    departure[0] - departure_start[0],
                    departure[1] - departure_start[1],
                ],
                inward,
            ) > 0.0
        );
        assert!(
            dot_xy(planar_forward(simulation.world().root_quaternion()), inward) > 0.5,
            "departure={departure:?} forward={:?} velocity={:?}",
            planar_forward(simulation.world().root_quaternion()),
            simulation.world().root_velocity(),
        );
        assert!(departure[2] > 20.0);
    }

    #[test]
    fn food_contact_disables_flight_only_after_grounding() {
        assert!(grounded_food_contact_blocks_flight(
            FlightMode::Grounded,
            true,
            BehaviorMode::Feed,
        ));
        assert!(!grounded_food_contact_blocks_flight(
            FlightMode::Cruise,
            true,
            BehaviorMode::Feed,
        ));
    }

    #[test]
    fn grounded_fly_walks_under_a_table_without_an_airborne_escape_override() {
        let mut simulation =
            SimulationStepper::new(DEFAULT_ASSETS_DIR, None::<&str>, 500.0, 0.0).unwrap();
        let initial_qpos = simulation.world().qpos()[0..7].to_vec();
        {
            let data = simulation.world_mut().data_mut();
            data.qpos_mut()[0] = initial_qpos[0] + 119.5;
            data.qpos_mut()[1] = initial_qpos[1] + 70.0;
            data.qvel_mut()[..6].fill(0.0);
            data.forward();
        }
        let start = simulation.world().root_position();
        let mut final_snapshot = simulation.snapshot();
        for _ in 0..500 {
            final_snapshot = simulation.step_window().unwrap();
            assert_eq!(final_snapshot.flight_mode, FlightMode::Grounded);
            assert!(!final_snapshot.flight_escape_active);
            assert_eq!(
                simulation.ground_navigation.escape_side(),
                crate::obstacle_avoidance::EscapeSide::None
            );
        }
        let displacement = (final_snapshot.root_position[0] - start[0])
            .hypot(final_snapshot.root_position[1] - start[1]);
        assert!(displacement > 1.0, "displacement {displacement}");
        let blocked = simulation
            .ground_navigation
            .update(crate::obstacle_avoidance::NavigationObservation {
                forward_clearance_mm: 1.5,
                up_clearance_mm: 50.0,
                ..Default::default()
            })
            .unwrap();
        assert!(blocked.obstacle_active && blocked.collision_reflex_active);
    }

    #[test]
    fn manual_grooming_waits_for_support_and_overrides_grounded_motion() {
        let mut simulation =
            SimulationStepper::new(DEFAULT_ASSETS_DIR, None::<&str>, 500.0, 0.5).unwrap();
        simulation.request_grooming();
        let mut grooming_snapshot = None;
        for _ in 0..5000 {
            let snapshot = simulation.step_window().unwrap();
            if snapshot.grooming_active {
                grooming_snapshot = Some(snapshot);
                break;
            }
        }
        let snapshot = grooming_snapshot.expect("manual grooming did not start");
        assert_eq!(snapshot.grooming_trigger, GroomingTrigger::Manual);
        assert_eq!(snapshot.grooming_support_leg_count, 4);
        assert_eq!(snapshot.flight_mode, FlightMode::Grounded);
        assert!(!snapshot.taste_active);
    }

    #[test]
    fn wingbeat_frequency_tracks_height_error_with_bounded_feedback() {
        assert_eq!(flight_frequency_scale(FlightMode::Grounded, 30.0, 0.0), 1.0);
        assert!(flight_frequency_scale(FlightMode::Takeoff, 20.0, 0.0) > 1.35);
        assert!(flight_frequency_scale(FlightMode::Cruise, -20.0, 0.0) < 1.2);
        assert_eq!(flight_frequency_scale(FlightMode::Landing, 0.0, 0.0), 1.25);
        assert_eq!(
            flight_frequency_scale(FlightMode::Takeoff, 100.0, -100.0),
            1.5
        );
    }

    #[test]
    fn airborne_overhang_escape_commands_a_bounded_descent() {
        assert_eq!(
            commanded_flight_height(FlightMode::Cruise, 44.0, 28.0, 5.0, true, true),
            14.0
        );
        assert_eq!(
            commanded_flight_height(FlightMode::Cruise, 44.0, 10.0, 5.0, true, true),
            5.0
        );
        assert_eq!(
            commanded_flight_height(FlightMode::Cruise, 44.0, 3.0, 5.0, true, true),
            5.0
        );
        assert_eq!(
            commanded_flight_height(FlightMode::Cruise, 44.0, 28.0, 5.0, false, false),
            44.0
        );
        assert_eq!(
            commanded_flight_height(FlightMode::Cruise, 44.0, 3.0, 5.0, true, false),
            5.0
        );
    }

    #[test]
    fn wall_takeoff_direction_points_into_the_room() {
        let half_extents = [300.0, 200.0, 210.0];
        assert_eq!(
            wall_inward_direction([298.0, 0.0, 80.0], half_extents),
            [-1.0, 0.0]
        );
        assert_eq!(
            wall_inward_direction([-298.0, 0.0, 80.0], half_extents),
            [1.0, 0.0]
        );
        assert_eq!(
            wall_inward_direction([0.0, 198.0, 80.0], half_extents),
            [0.0, -1.0]
        );
        assert_eq!(
            wall_inward_direction([0.0, -198.0, 80.0], half_extents),
            [0.0, 1.0]
        );
        let corner = wall_inward_direction([298.0, 198.0, 80.0], half_extents);
        assert!((corner[0] + std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert!((corner[1] + std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert_eq!(planar_wall_clearance([296.0, 0.0, 80.0], half_extents), 4.0);
        assert_eq!(
            planar_wall_clearance([0.0, -198.0, 80.0], half_extents),
            2.0
        );
    }

    #[test]
    fn latched_wall_escape_replans_when_another_wall_blocks_its_heading() {
        let room = [300.0, 220.0, 110.0];
        let mut state = WallEscapeState::default();
        let mut observation = WallEscapeObservation {
            mode: FlightMode::Cruise,
            wall_takeoff: false,
            wall_support_leg_count: 0,
            position_mm: [250.0, 217.0, 180.0],
            room_half_extents_mm: room,
            forward_xy: [0.0, -1.0],
            planar_velocity_mm_s: [0.0, -100.0],
        };
        state.update(observation);
        assert_eq!(state.direction_xy, [0.0, -1.0]);
        observation.position_mm = [296.0, 100.0, 180.0];
        state.update(observation);
        assert_eq!(state.direction_xy, [-1.0, 0.0]);
        observation.position_mm = [296.0, -217.0, 180.0];
        state.direction_xy = [0.0, -1.0];
        state.update(observation);
        assert!(state.direction_xy[0] < 0.0 && state.direction_xy[1] > 0.0);
        let corner_direction = state.direction_xy;
        observation.position_mm = [295.0, -217.0, 180.0];
        state.update(observation);
        assert_eq!(state.direction_xy, corner_direction);
        assert!(state.latched);
    }

    #[test]
    fn wall_escape_releases_only_after_clear_aligned_inward_flight() {
        let room = [300.0, 220.0, 110.0];
        let mut state = WallEscapeState::default();

        state.update(WallEscapeObservation {
            mode: FlightMode::Cruise,
            wall_takeoff: false,
            wall_support_leg_count: 0,
            position_mm: [296.0, 0.0, 80.0],
            room_half_extents_mm: room,
            forward_xy: [1.0, 0.0],
            planar_velocity_mm_s: [200.0, 0.0],
        });
        assert!(state.latched);

        state.update(WallEscapeObservation {
            mode: FlightMode::Cruise,
            wall_takeoff: false,
            wall_support_leg_count: 0,
            position_mm: [279.0, 0.0, 80.0],
            room_half_extents_mm: room,
            forward_xy: [1.0, 0.0],
            planar_velocity_mm_s: [200.0, 0.0],
        });
        assert!(state.latched);

        state.update(WallEscapeObservation {
            mode: FlightMode::Cruise,
            wall_takeoff: false,
            wall_support_leg_count: 0,
            position_mm: [279.0, 0.0, 80.0],
            room_half_extents_mm: room,
            forward_xy: [-1.0, 0.0],
            planar_velocity_mm_s: [200.0, 0.0],
        });
        assert!(state.latched);

        state.update(WallEscapeObservation {
            mode: FlightMode::Cruise,
            wall_takeoff: false,
            wall_support_leg_count: 0,
            position_mm: [279.0, 0.0, 80.0],
            room_half_extents_mm: room,
            forward_xy: [-1.0, 0.0],
            planar_velocity_mm_s: [-25.0, 0.0],
        });
        assert!(state.latched);

        for _ in 1..WALL_ESCAPE_RELEASE_DWELL_WINDOWS {
            state.update(WallEscapeObservation {
                mode: FlightMode::Cruise,
                wall_takeoff: false,
                wall_support_leg_count: 0,
                position_mm: [279.0, 0.0, 80.0],
                room_half_extents_mm: room,
                forward_xy: [-1.0, 0.0],
                planar_velocity_mm_s: [-25.0, 0.0],
            });
        }
        assert!(!state.latched);
        assert!(state.release_hold_windows > 0);
    }

    #[test]
    fn wall_escape_turns_toward_its_latched_inward_heading() {
        assert_eq!(wall_escape_speed_scale([1.0, 0.0], [-1.0, 0.0]), 0.0);
        assert_eq!(wall_escape_speed_scale([0.0, 1.0], [-1.0, 0.0]), 0.0);
        assert_eq!(wall_escape_speed_scale([-1.0, 0.0], [-1.0, 0.0]), 1.0);
    }

    #[test]
    fn full_brain_walking_uses_neural_turn_except_during_explicit_arbitration() {
        assert_eq!(
            full_brain_walking_turn(BehaviorMode::Explore, 0.8, 0.4, 0.5),
            0.2
        );
        assert_eq!(
            full_brain_walking_turn(BehaviorMode::TrackOdor, -0.8, -0.4, 0.5),
            -0.2
        );
        assert_eq!(
            full_brain_walking_turn(BehaviorMode::Feed, 0.8, 0.4, 0.5),
            0.0
        );
        assert_eq!(
            full_brain_walking_turn(BehaviorMode::DepartFood, 0.8, 0.4, 0.5),
            0.8
        );
    }

    #[test]
    fn full_brain_walking_uses_common_neural_activity_for_forward_drive() {
        assert_eq!(
            full_brain_walking_forward(BehaviorMode::Explore, 0.8, 0.4),
            0.4
        );
        assert_eq!(
            full_brain_walking_forward(BehaviorMode::TrackOdor, 0.9, 0.3),
            0.3
        );
        assert_eq!(
            full_brain_walking_forward(BehaviorMode::Feed, 0.8, 0.4),
            0.0
        );
        assert_eq!(
            full_brain_walking_forward(BehaviorMode::DepartFood, 0.8, 0.4),
            0.8
        );
    }

    #[test]
    fn optic_flow_control_changes_altitude_only_with_valid_clearance() {
        assert!(optic_flow_altitude_control(300.0, 28.0, 28.0, false) > 0.9);
        assert_eq!(optic_flow_altitude_control(300.0, 28.0, 28.0, true), 0.0);
        assert_eq!(optic_flow_altitude_control(30.0, 70.0, 70.0, false), 0.0);
        assert!(optic_flow_altitude_control(80.0, 80.0, 100.0, false) < 0.0);
    }
}
