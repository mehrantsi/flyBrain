use std::path::Path;

use anyhow::{Context, Result, bail};
use mujoco_rs::prelude::MjtObj;
use serde::{Deserialize, Serialize};

use crate::aerodynamics::{AerodynamicsConfig, Vec3, WingStrip, WingbeatGenerator};
use crate::world::{MuJoCoWorld, WING_ACTUATOR_COUNT};

const ENGINEERED_FLIGHT_TARGET_PITCH_RAD: f64 = -0.8290313946973065;
const ENGINEERED_BODY_PITCH_FEEDFORWARD_TORQUE_G_MM2_S2: f64 = -7.0;
const ENGINEERED_BODY_POSITION_GAIN_G_MM2_S2: f64 = 40.0;
const ENGINEERED_BODY_RATE_GAIN_G_MM2_S2: f64 = 0.04;
const ENGINEERED_BODY_YAW_RATE_GAIN_G_MM2_S2: f64 = 1.0;
const ENGINEERED_BODY_HEADING_GAIN_PER_S: f64 = 4.0;
const ENGINEERED_BODY_MAX_YAW_RATE_RAD_S: f64 = 2.0;
const ENGINEERED_BODY_MAX_TORQUE_G_MM2_S2: f64 = 60.0;
const FLYBODY_VISION_TASK_TARGET_HORIZONTAL_SPEED_MM_S: f64 = 300.0;
const ENGINEERED_BODY_VELOCITY_GAIN_PER_S: f64 = 200.0;
// Engineering choice: keep the I time at 100 ms as P is calibrated, so
// disturbance rejection remains fast without changing the existing P loop.
const ENGINEERED_BODY_VELOCITY_INTEGRAL_TIME_S: f64 = 0.1;
const ENGINEERED_BODY_MAX_HORIZONTAL_FORCE_WEIGHT: f64 = 2.0;
const ENGINEERED_BODY_ALTITUDE_POSITION_GAIN_PER_S2: f64 = 600.0;
const ENGINEERED_BODY_ALTITUDE_RATE_GAIN_PER_S: f64 = 50.0;
const ENGINEERED_BODY_ALTITUDE_INTEGRAL_TIME_S: f64 = 0.2;
const ENGINEERED_BODY_MAX_VERTICAL_FORCE_WEIGHT: f64 = 0.8;
const RETRACTED_WING_CONTROLS_RAD: [f64; WING_ACTUATOR_COUNT] = [1.5, -1.0, 0.7, 1.5, -1.0, 0.7];
const WING_RETRACTION_RATE_RAD_S: f64 = 100.0;
const WALL_LANDING_FOOT_CLEARANCE_MM: f64 = 1.1;
const WALL_LANDING_ROTATION_CLEARANCE_MM: f64 = 5.0;
const WALL_LANDING_APPROACH_GAIN_PER_S: f64 = 20.0;
const WALL_LANDING_APPROACH_SPEED_MM_S: f64 = 12.0;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlightDynamicsParameters {
    pub target_pitch_rad: f64,
    pub pitch_feedforward_torque_g_mm2_s2: f64,
    pub attitude_position_gain_g_mm2_s2: f64,
    pub attitude_rate_gain_g_mm2_s2: f64,
    pub yaw_rate_gain_g_mm2_s2: f64,
    pub maximum_yaw_rate_rad_s: f64,
    pub maximum_torque_g_mm2_s2: f64,
    pub target_horizontal_speed_mm_s: f64,
    pub velocity_gain_per_s: f64,
    pub maximum_horizontal_force_weight: f64,
    pub altitude_position_gain_per_s2: f64,
    pub altitude_rate_gain_per_s: f64,
    pub maximum_vertical_force_weight: f64,
}

impl Default for FlightDynamicsParameters {
    fn default() -> Self {
        Self {
            target_pitch_rad: ENGINEERED_FLIGHT_TARGET_PITCH_RAD,
            pitch_feedforward_torque_g_mm2_s2: ENGINEERED_BODY_PITCH_FEEDFORWARD_TORQUE_G_MM2_S2,
            attitude_position_gain_g_mm2_s2: ENGINEERED_BODY_POSITION_GAIN_G_MM2_S2,
            attitude_rate_gain_g_mm2_s2: ENGINEERED_BODY_RATE_GAIN_G_MM2_S2,
            yaw_rate_gain_g_mm2_s2: ENGINEERED_BODY_YAW_RATE_GAIN_G_MM2_S2,
            maximum_yaw_rate_rad_s: ENGINEERED_BODY_MAX_YAW_RATE_RAD_S,
            maximum_torque_g_mm2_s2: ENGINEERED_BODY_MAX_TORQUE_G_MM2_S2,
            target_horizontal_speed_mm_s: FLYBODY_VISION_TASK_TARGET_HORIZONTAL_SPEED_MM_S,
            velocity_gain_per_s: ENGINEERED_BODY_VELOCITY_GAIN_PER_S,
            maximum_horizontal_force_weight: ENGINEERED_BODY_MAX_HORIZONTAL_FORCE_WEIGHT,
            altitude_position_gain_per_s2: ENGINEERED_BODY_ALTITUDE_POSITION_GAIN_PER_S2,
            altitude_rate_gain_per_s: ENGINEERED_BODY_ALTITUDE_RATE_GAIN_PER_S,
            maximum_vertical_force_weight: ENGINEERED_BODY_MAX_VERTICAL_FORCE_WEIGHT,
        }
    }
}

impl FlightDynamicsParameters {
    pub fn validate(self) -> Result<Self> {
        if !self.target_pitch_rad.is_finite()
            || !(-std::f64::consts::FRAC_PI_2..=std::f64::consts::FRAC_PI_2)
                .contains(&self.target_pitch_rad)
            || !self.pitch_feedforward_torque_g_mm2_s2.is_finite()
            || [
                self.attitude_position_gain_g_mm2_s2,
                self.attitude_rate_gain_g_mm2_s2,
                self.yaw_rate_gain_g_mm2_s2,
                self.maximum_yaw_rate_rad_s,
                self.maximum_torque_g_mm2_s2,
                self.target_horizontal_speed_mm_s,
                self.velocity_gain_per_s,
                self.maximum_horizontal_force_weight,
                self.altitude_position_gain_per_s2,
                self.altitude_rate_gain_per_s,
                self.maximum_vertical_force_weight,
            ]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
            || self.maximum_horizontal_force_weight > 2.0
            || self.maximum_vertical_force_weight > 2.0
        {
            bail!("flight dynamics parameters are invalid")
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WallLandingTarget {
    pub surface_point_mm: [f64; 3],
    pub inward_xy: [f64; 2],
}

impl WallLandingTarget {
    pub fn orientation(self) -> [f64; 4] {
        let half_yaw = (-self.inward_xy[1]).atan2(-self.inward_xy[0]) * 0.5;
        let c = half_yaw.cos() * std::f64::consts::FRAC_1_SQRT_2;
        let s = half_yaw.sin() * std::f64::consts::FRAC_1_SQRT_2;
        [c, s, -c, s]
    }

    pub fn alignment(self, quaternion: [f64; 4]) -> f64 {
        let [w, x, y, z] = quaternion;
        2.0 * (x * z + w * y) * self.inward_xy[0] + 2.0 * (y * z - w * x) * self.inward_xy[1]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlightCommand {
    pub enabled: bool,
    pub amplitude: f64,
    pub steering: f64,
    pub wing_steering_scale: f64,
    pub horizontal_speed_scale: f64,
    pub heading_target_xy: Option<[f64; 2]>,
    pub planar_velocity_direction: Option<[f64; 2]>,
    pub altitude_target_mm: Option<f64>,
    pub body_pitch_target_rad: Option<f64>,
    pub wall_landing: Option<WallLandingTarget>,
    pub frequency_scale: f64,
    pub pitch_bias_rad: f64,
    pub roll_bias_rad: f64,
    pub differential_pitch_rad: f64,
    pub differential_roll_rad: f64,
}

impl Default for FlightCommand {
    fn default() -> Self {
        Self {
            enabled: false,
            amplitude: 0.0,
            steering: 0.0,
            wing_steering_scale: 1.0,
            horizontal_speed_scale: 1.0,
            heading_target_xy: None,
            planar_velocity_direction: None,
            altitude_target_mm: None,
            body_pitch_target_rad: None,
            wall_landing: None,
            frequency_scale: 1.0,
            pitch_bias_rad: 0.0,
            roll_bias_rad: 0.0,
            differential_pitch_rad: 0.0,
            differential_roll_rad: 0.0,
        }
    }
}

impl FlightCommand {
    pub fn validate(self) -> Result<Self> {
        if !self.amplitude.is_finite()
            || !(0.0..=1.0).contains(&self.amplitude)
            || !self.steering.is_finite()
            || !(-1.0..=1.0).contains(&self.steering)
            || !self.wing_steering_scale.is_finite()
            || !(0.0..=1.0).contains(&self.wing_steering_scale)
            || !self.horizontal_speed_scale.is_finite()
            || !(0.0..=1.0).contains(&self.horizontal_speed_scale)
            || self.heading_target_xy.is_some_and(|direction| {
                direction.iter().any(|value| !value.is_finite())
                    || direction[0].hypot(direction[1]) <= 1e-9
            })
            || self.planar_velocity_direction.is_some_and(|direction| {
                direction.iter().any(|value| !value.is_finite())
                    || direction[0].hypot(direction[1]) <= 1e-9
            })
            || self
                .altitude_target_mm
                .is_some_and(|height| !height.is_finite() || height < 0.0)
            || !self.frequency_scale.is_finite()
            || self.body_pitch_target_rad.is_some_and(|pitch| {
                !pitch.is_finite() || pitch.abs() > std::f64::consts::FRAC_PI_2
            })
            || self.wall_landing.is_some_and(|target| {
                target
                    .surface_point_mm
                    .iter()
                    .chain(target.inward_xy.iter())
                    .any(|v| !v.is_finite())
                    || (target.inward_xy[0].hypot(target.inward_xy[1]) - 1.0).abs() > 1e-6
            })
            || !(0.5..=1.5).contains(&self.frequency_scale)
            || !self.pitch_bias_rad.is_finite()
            || !(-1.0..=1.0).contains(&self.pitch_bias_rad)
            || !self.roll_bias_rad.is_finite()
            || !(-1.0..=1.0).contains(&self.roll_bias_rad)
            || !self.differential_pitch_rad.is_finite()
            || !(-1.0..=1.0).contains(&self.differential_pitch_rad)
            || !self.differential_roll_rad.is_finite()
            || !(-1.0..=1.0).contains(&self.differential_roll_rad)
        {
            bail!("flight command amplitude or steering is invalid")
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WingFlightTelemetry {
    pub side: String,
    pub force_g_mm_s2: Vec3,
    pub moment_g_mm2_s2: Vec3,
    pub peak_strip_speed_mm_s: f64,
    pub mean_angle_of_attack_rad: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlightTelemetry {
    pub enabled: bool,
    pub controls_rad: [f64; WING_ACTUATOR_COUNT],
    pub wings: [WingFlightTelemetry; 2],
    pub total_force_g_mm_s2: Vec3,
    pub total_moment_g_mm2_s2: Vec3,
    pub root_qfrc_fluid: [f64; 6],
    pub engineered_body_stabilizer_enabled: bool,
    pub engineered_body_velocity_force_g_mm_s2: Vec3,
    pub engineered_body_stabilizer_torque_g_mm2_s2: Vec3,
    pub engineered_body_target_pitch_rad: f64,
    pub weight_g_mm_s2: f64,
    pub vertical_force_to_weight: f64,
}

impl FlightTelemetry {
    fn set_fluid_force(&mut self, force: [f64; 6]) {
        self.root_qfrc_fluid = force;
        self.total_force_g_mm_s2 = Vec3::new(force[0], force[1], force[2]);
        self.total_moment_g_mm2_s2 = Vec3::new(force[3], force[4], force[5]);
        self.vertical_force_to_weight = self.total_force_g_mm_s2.z() / self.weight_g_mm_s2;
    }
}

pub struct FlightRuntime {
    config: AerodynamicsConfig,
    dynamics: FlightDynamicsParameters,
    generator: WingbeatGenerator,
    wing_indices: [usize; 2],
    body_ids: [usize; 2],
    root_body_id: usize,
    body_mass_g: f64,
    horizontal_velocity_integral_mm: [f64; 2],
    last_horizontal_integral_time_s: Option<f64>,
    altitude_position_error_integral_mm_s: f64,
    last_altitude_integral_time_s: Option<f64>,
    wingbeat_phase_time_s: f64,
    last_wingbeat_time_s: Option<f64>,
    last_wingbeat_frequency_scale: Option<f64>,
    last_controls: [f64; WING_ACTUATOR_COUNT],
    last_control_time_s: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct FlightStabilizer {
    pub amplitude: f64,
    pub target_pitch_rad: f64,
    pub pitch_bias_rad: f64,
    pub pitch_position_gain: f64,
    pub pitch_rate_gain: f64,
    pub roll_position_gain: f64,
    pub roll_rate_gain: f64,
}

impl Default for FlightStabilizer {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            target_pitch_rad: ENGINEERED_FLIGHT_TARGET_PITCH_RAD,
            pitch_bias_rad: 0.0,
            pitch_position_gain: 0.2,
            pitch_rate_gain: -0.00005,
            roll_position_gain: 0.08,
            roll_rate_gain: 0.0001,
        }
    }
}

impl FlightStabilizer {
    pub fn from_dynamics(parameters: FlightDynamicsParameters) -> Result<Self> {
        Ok(Self {
            target_pitch_rad: parameters.validate()?.target_pitch_rad,
            ..Self::default()
        })
    }

    pub fn command(
        self,
        root_quaternion: [f64; 4],
        root_velocity: [f64; 6],
        amplitude_scale: f64,
    ) -> Result<FlightCommand> {
        self.command_with_base(
            root_quaternion,
            root_velocity,
            FlightCommand {
                enabled: self.amplitude > 0.0,
                amplitude: self.amplitude,
                steering: 0.0,
                wing_steering_scale: 1.0,
                horizontal_speed_scale: 1.0,
                heading_target_xy: None,
                planar_velocity_direction: None,
                altitude_target_mm: None,
                body_pitch_target_rad: None,
                wall_landing: None,
                frequency_scale: 1.0,
                pitch_bias_rad: 0.0,
                roll_bias_rad: 0.0,
                differential_pitch_rad: 0.0,
                differential_roll_rad: 0.0,
            },
            amplitude_scale,
        )
    }

    pub fn command_with_base(
        self,
        root_quaternion: [f64; 4],
        root_velocity: [f64; 6],
        base_command: FlightCommand,
        amplitude_scale: f64,
    ) -> Result<FlightCommand> {
        self.command_with_base_internal(
            root_quaternion,
            root_velocity,
            base_command,
            amplitude_scale,
            None,
        )
    }

    pub fn command_with_base_limited(
        self,
        root_quaternion: [f64; 4],
        root_velocity: [f64; 6],
        base_command: FlightCommand,
        amplitude_scale: f64,
        config: &AerodynamicsConfig,
    ) -> Result<FlightCommand> {
        self.command_with_base_internal(
            root_quaternion,
            root_velocity,
            base_command,
            amplitude_scale,
            Some(config),
        )
    }

    fn command_with_base_internal(
        self,
        root_quaternion: [f64; 4],
        root_velocity: [f64; 6],
        base_command: FlightCommand,
        amplitude_scale: f64,
        config: Option<&AerodynamicsConfig>,
    ) -> Result<FlightCommand> {
        if root_quaternion.iter().any(|value| !value.is_finite())
            || root_velocity.iter().any(|value| !value.is_finite())
            || !amplitude_scale.is_finite()
            || !(0.0..=1.0).contains(&amplitude_scale)
        {
            bail!("flight stabilizer input is invalid")
        }
        let base_command = base_command.validate()?;
        if base_command.wall_landing.is_some() {
            let command = FlightCommand {
                enabled: base_command.enabled && amplitude_scale > 0.0,
                amplitude: base_command.amplitude * amplitude_scale,
                ..base_command
            };
            return config.map_or(Ok(command), |config| {
                project_attitude_feedback(command, config)
            });
        }
        let [w, x, y, z] = root_quaternion;
        let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
        let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        let heading_roll_rate = yaw.cos() * root_velocity[0] + yaw.sin() * root_velocity[1];
        let heading_pitch_rate = -yaw.sin() * root_velocity[0] + yaw.cos() * root_velocity[1];
        let pitch_error = base_command
            .body_pitch_target_rad
            .unwrap_or(self.target_pitch_rad)
            - pitch;
        let roll_error = -roll;
        let command = FlightCommand {
            enabled: base_command.enabled && amplitude_scale > 0.0,
            amplitude: base_command.amplitude * amplitude_scale,
            steering: base_command.steering,
            wing_steering_scale: base_command.wing_steering_scale,
            horizontal_speed_scale: base_command.horizontal_speed_scale,
            heading_target_xy: base_command.heading_target_xy,
            planar_velocity_direction: base_command.planar_velocity_direction,
            altitude_target_mm: base_command.altitude_target_mm,
            body_pitch_target_rad: base_command.body_pitch_target_rad,
            wall_landing: base_command.wall_landing,
            frequency_scale: base_command.frequency_scale,
            pitch_bias_rad: (base_command.pitch_bias_rad
                + self.pitch_bias_rad
                + self.pitch_position_gain * pitch_error
                + self.pitch_rate_gain * heading_pitch_rate),
            roll_bias_rad: base_command.roll_bias_rad,
            differential_pitch_rad: (base_command.differential_pitch_rad
                + self.roll_position_gain * roll_error
                - self.roll_rate_gain * heading_roll_rate)
                .clamp(-1.0, 1.0),
            differential_roll_rad: base_command.differential_roll_rad,
        }
        .validate()?;
        if let Some(config) = config {
            project_attitude_feedback(command, config)
        } else {
            Ok(command)
        }
    }
}

fn project_attitude_feedback(
    command: FlightCommand,
    config: &AerodynamicsConfig,
) -> Result<FlightCommand> {
    let (pitch_bias_rad, differential_pitch_rad) = project_axis_feedback(
        command.pitch_bias_rad,
        command.differential_pitch_rad,
        command.amplitude,
        "pitch",
        config,
    )?;
    let (roll_bias_rad, differential_roll_rad) = project_axis_feedback(
        command.roll_bias_rad,
        command.differential_roll_rad,
        command.amplitude,
        "roll",
        config,
    )?;
    Ok(FlightCommand {
        pitch_bias_rad,
        roll_bias_rad,
        differential_pitch_rad,
        differential_roll_rad,
        ..command
    })
}

fn project_axis_feedback(
    symmetric_bias_rad: f64,
    differential_bias_rad: f64,
    amplitude: f64,
    axis: &str,
    config: &AerodynamicsConfig,
) -> Result<(f64, f64)> {
    let center = *config
        .wingbeat
        .center_rad_by_axis
        .get(axis)
        .with_context(|| format!("wingbeat center is missing {axis}"))?;
    let waveform_amplitude = *config
        .wingbeat
        .amplitude_rad_by_axis
        .get(axis)
        .with_context(|| format!("wingbeat amplitude is missing {axis}"))?;
    let range = *config
        .wingbeat
        .joint_ranges_rad
        .get(axis)
        .with_context(|| format!("wingbeat joint range is missing {axis}"))?;
    let base_low = center - amplitude * waveform_amplitude;
    let base_high = center + amplitude * waveform_amplitude;
    let bias_low = range[0] - base_low;
    let bias_high = range[1] - base_high;
    if bias_low > bias_high {
        bail!("wingbeat amplitude leaves no admissible {axis} feedback interval")
    }
    let maximum_differential = (bias_high - symmetric_bias_rad)
        .min(symmetric_bias_rad - bias_low)
        .max(0.0);
    let differential = differential_bias_rad.clamp(-maximum_differential, maximum_differential);
    let symmetric_low = bias_low + differential.abs();
    let symmetric_high = bias_high - differential.abs();
    Ok((
        symmetric_bias_rad.clamp(symmetric_low, symmetric_high),
        differential,
    ))
}

impl FlightRuntime {
    pub fn new(assets: impl AsRef<Path>, world: &MuJoCoWorld) -> Result<Self> {
        Self::new_with_parameters(assets, world, FlightDynamicsParameters::default())
    }

    pub fn new_with_parameters(
        assets: impl AsRef<Path>,
        world: &MuJoCoWorld,
        dynamics: FlightDynamicsParameters,
    ) -> Result<Self> {
        let dynamics = dynamics.validate()?;
        let config = AerodynamicsConfig::load(assets)?;
        let generator = config.wingbeat_generator()?;
        let wing_indices = [wing_index(&config, "left")?, wing_index(&config, "right")?];
        let body_ids = [
            aerodynamic_body_id(&config, world, wing_indices[0])?,
            aerodynamic_body_id(&config, world, wing_indices[1])?,
        ];
        let root_body_id = world
            .model()
            .name_to_id(MjtObj::mjOBJ_BODY, "fly/c_thorax")
            .context("model is missing free root body fly/c_thorax")?;
        for wing_index in wing_indices {
            for actuator in &config.wings[wing_index].actuator_names {
                if world
                    .model()
                    .name_to_id(MjtObj::mjOBJ_ACTUATOR, actuator)
                    .is_none()
                {
                    bail!("model is missing aerodynamic actuator {actuator}")
                }
            }
        }
        if config.uses_mujoco_ellipsoid() {
            validate_mujoco_fluid_geoms(&config, world)?;
        }
        let body_mass_g = world.model().body_mass().iter().sum::<f64>();
        if !body_mass_g.is_finite() || body_mass_g <= 0.0 {
            bail!("model total body mass is invalid")
        }
        Ok(Self {
            config,
            dynamics,
            generator,
            wing_indices,
            body_ids,
            root_body_id,
            body_mass_g,
            horizontal_velocity_integral_mm: [0.0; 2],
            last_horizontal_integral_time_s: None,
            altitude_position_error_integral_mm_s: 0.0,
            last_altitude_integral_time_s: None,
            wingbeat_phase_time_s: 0.0,
            last_wingbeat_time_s: None,
            last_wingbeat_frequency_scale: None,
            last_controls: RETRACTED_WING_CONTROLS_RAD,
            last_control_time_s: None,
        })
    }

    pub fn config(&self) -> &AerodynamicsConfig {
        &self.config
    }

    pub fn dynamics(&self) -> FlightDynamicsParameters {
        self.dynamics
    }

    pub fn apply(
        &mut self,
        world: &mut MuJoCoWorld,
        command: FlightCommand,
        air_velocity_mm_s: [f64; 3],
    ) -> Result<FlightTelemetry> {
        self.apply_internal(world, command, air_velocity_mm_s, true)
    }

    pub fn advance(
        &mut self,
        world: &mut MuJoCoWorld,
        command: FlightCommand,
        air_velocity_mm_s: [f64; 3],
    ) -> Result<FlightTelemetry> {
        let mut telemetry = self.apply_internal(world, command, air_velocity_mm_s, false)?;
        world.step()?;
        if self.config.uses_mujoco_ellipsoid() {
            telemetry.set_fluid_force(root_qfrc_fluid(world)?);
        }
        Ok(telemetry)
    }

    fn apply_internal(
        &mut self,
        world: &mut MuJoCoWorld,
        command: FlightCommand,
        air_velocity_mm_s: [f64; 3],
        refresh_fluid_force: bool,
    ) -> Result<FlightTelemetry> {
        let command = command.validate()?;
        if air_velocity_mm_s.iter().any(|value| !value.is_finite()) {
            bail!("air velocity must contain finite values")
        }
        let controls = self.controls(world.time(), command)?;
        world.set_wing_controls(controls)?;
        let engineered_body_stabilizer_enabled = command.enabled && command.amplitude > 0.0;
        if !engineered_body_stabilizer_enabled {
            self.reset_horizontal_velocity_integral();
            self.reset_altitude_position_error_integral();
        }
        let engineered_body_stabilizer_torque = if engineered_body_stabilizer_enabled {
            compute_engineered_body_stabilizer_torque(world, command, self.dynamics)
        } else {
            Vec3::ZERO
        };
        let engineered_body_velocity_force = if engineered_body_stabilizer_enabled {
            self.compute_engineered_body_velocity_force(
                world,
                command.horizontal_speed_scale,
                command.planar_velocity_direction,
                command.altitude_target_mm,
                command.wall_landing,
            )
        } else {
            Vec3::ZERO
        };
        world.data_mut().xfrc_applied_mut()[self.root_body_id] = [
            engineered_body_velocity_force.x(),
            engineered_body_velocity_force.y(),
            engineered_body_velocity_force.z(),
            engineered_body_stabilizer_torque.x(),
            engineered_body_stabilizer_torque.y(),
            engineered_body_stabilizer_torque.z(),
        ];
        let (wing_telemetry, root_qfrc_fluid_g_mm_s2) = if self.config.uses_mujoco_ellipsoid() {
            {
                let applied = world.data_mut().xfrc_applied_mut();
                for body_id in self.body_ids {
                    applied[body_id] = [0.0; 6];
                }
            }
            if refresh_fluid_force {
                world.data_mut().forward();
            }
            (
                [self.zero_wing(0), self.zero_wing(1)],
                if refresh_fluid_force {
                    root_qfrc_fluid(world)?
                } else {
                    [0.0; 6]
                },
            )
        } else if command.enabled {
            (
                [
                    self.wing_force(world, 0, air_velocity_mm_s.into())?,
                    self.wing_force(world, 1, air_velocity_mm_s.into())?,
                ],
                [0.0; 6],
            )
        } else {
            ([self.zero_wing(0), self.zero_wing(1)], [0.0; 6])
        };
        if !self.config.uses_mujoco_ellipsoid() {
            let applied = world.data_mut().xfrc_applied_mut();
            for (body_id, wing) in self.body_ids.into_iter().zip(&wing_telemetry) {
                applied[body_id] = [
                    wing.force_g_mm_s2.x(),
                    wing.force_g_mm_s2.y(),
                    wing.force_g_mm_s2.z(),
                    wing.moment_g_mm2_s2.x(),
                    wing.moment_g_mm2_s2.y(),
                    wing.moment_g_mm2_s2.z(),
                ];
            }
        }
        let total_force_g_mm_s2 = if self.config.uses_mujoco_ellipsoid() {
            Vec3::new(
                root_qfrc_fluid_g_mm_s2[0],
                root_qfrc_fluid_g_mm_s2[1],
                root_qfrc_fluid_g_mm_s2[2],
            )
        } else {
            wing_telemetry
                .iter()
                .fold(Vec3::ZERO, |total, wing| total + wing.force_g_mm_s2)
        };
        let total_moment_g_mm2_s2 = if self.config.uses_mujoco_ellipsoid() {
            Vec3::new(
                root_qfrc_fluid_g_mm_s2[3],
                root_qfrc_fluid_g_mm_s2[4],
                root_qfrc_fluid_g_mm_s2[5],
            )
        } else {
            wing_telemetry
                .iter()
                .fold(Vec3::ZERO, |total, wing| total + wing.moment_g_mm2_s2)
        };
        let weight_g_mm_s2 = self.body_mass_g * 9_810.0;
        Ok(FlightTelemetry {
            enabled: command.enabled,
            controls_rad: controls,
            wings: wing_telemetry,
            total_force_g_mm_s2,
            total_moment_g_mm2_s2,
            root_qfrc_fluid: root_qfrc_fluid_g_mm_s2,
            engineered_body_stabilizer_enabled,
            engineered_body_velocity_force_g_mm_s2: engineered_body_velocity_force,
            engineered_body_stabilizer_torque_g_mm2_s2: engineered_body_stabilizer_torque,
            engineered_body_target_pitch_rad: self.dynamics.target_pitch_rad,
            weight_g_mm_s2,
            vertical_force_to_weight: total_force_g_mm_s2.z() / weight_g_mm_s2,
        })
    }

    fn controls(
        &mut self,
        time_seconds: f64,
        command: FlightCommand,
    ) -> Result<[f64; WING_ACTUATOR_COUNT]> {
        if !time_seconds.is_finite() {
            bail!("flight time must be finite")
        }
        let elapsed = self
            .last_control_time_s
            .map_or(0.0, |last| time_seconds - last);
        if elapsed < 0.0 {
            self.last_controls = RETRACTED_WING_CONTROLS_RAD;
        }
        self.last_control_time_s = Some(time_seconds);
        if !command.enabled {
            self.reset_wingbeat_phase();
            let maximum_change = WING_RETRACTION_RATE_RAD_S * elapsed.max(0.0);
            for (control, target) in self
                .last_controls
                .iter_mut()
                .zip(RETRACTED_WING_CONTROLS_RAD)
            {
                *control = target.clamp(*control - maximum_change, *control + maximum_change);
            }
            return Ok(self.last_controls);
        }
        let phase_time_seconds = self.wingbeat_phase(time_seconds, command.frequency_scale);
        let generated = self.generator.command(phase_time_seconds)?;
        let center = [
            self.config.wingbeat.center_rad_by_axis["yaw"],
            self.config.wingbeat.center_rad_by_axis["pitch"],
            self.config.wingbeat.center_rad_by_axis["roll"],
        ];
        let steering = 0.02 * command.wing_steering_scale * command.steering;
        let left_scale = command.amplitude * (1.0 - steering);
        let right_scale = command.amplitude * (1.0 + steering);
        let left: [f64; 3] = std::array::from_fn(|axis| {
            let scale = if axis == 1 {
                command.amplitude
            } else {
                left_scale
            };
            center[axis] + scale * (generated.left[axis] - center[axis])
        });
        let right: [f64; 3] = std::array::from_fn(|axis| {
            let scale = if axis == 1 {
                command.amplitude
            } else {
                right_scale
            };
            center[axis] + scale * (generated.right[axis] - center[axis])
        });
        let mut controls = [left[0], left[1], left[2], right[0], right[1], right[2]];
        controls[1] += command.pitch_bias_rad + command.differential_pitch_rad;
        controls[4] += command.pitch_bias_rad - command.differential_pitch_rad;
        controls[2] += command.roll_bias_rad + command.differential_roll_rad;
        controls[5] += command.roll_bias_rad - command.differential_roll_rad;
        for (axis, control) in controls.iter().enumerate() {
            let range =
                self.config.wingbeat.joint_ranges_rad[&self.config.wingbeat.joint_order[axis % 3]];
            if *control < range[0] || *control > range[1] {
                bail!(
                    "flight command axis {} value {} is outside [{}, {}] (amplitude {}, steering {}, pitch_bias {}, roll_bias {}, differential_pitch {}, differential_roll {})",
                    axis,
                    control,
                    range[0],
                    range[1],
                    command.amplitude,
                    command.steering,
                    command.pitch_bias_rad,
                    command.roll_bias_rad,
                    command.differential_pitch_rad,
                    command.differential_roll_rad,
                )
            }
        }
        self.last_controls = controls;
        Ok(controls)
    }

    fn reset_wingbeat_phase(&mut self) {
        self.wingbeat_phase_time_s = 0.0;
        self.last_wingbeat_time_s = None;
        self.last_wingbeat_frequency_scale = None;
    }

    fn wingbeat_phase(&mut self, time_seconds: f64, frequency_scale: f64) -> f64 {
        let Some(last_time_seconds) = self.last_wingbeat_time_s else {
            self.last_wingbeat_time_s = Some(time_seconds);
            self.last_wingbeat_frequency_scale = Some(frequency_scale);
            return self.wingbeat_phase_time_s;
        };
        let elapsed_seconds = time_seconds - last_time_seconds;
        if elapsed_seconds < 0.0 {
            self.wingbeat_phase_time_s = 0.0;
            self.last_wingbeat_time_s = Some(time_seconds);
            self.last_wingbeat_frequency_scale = Some(frequency_scale);
            return self.wingbeat_phase_time_s;
        }
        if elapsed_seconds > 0.0 {
            self.wingbeat_phase_time_s += elapsed_seconds
                * self
                    .last_wingbeat_frequency_scale
                    .unwrap_or(frequency_scale);
        }
        self.last_wingbeat_time_s = Some(time_seconds);
        self.last_wingbeat_frequency_scale = Some(frequency_scale);
        self.wingbeat_phase_time_s
    }

    fn compute_engineered_body_velocity_force(
        &mut self,
        world: &MuJoCoWorld,
        horizontal_speed_scale: f64,
        planar_direction_override: Option<[f64; 2]>,
        altitude_target_mm: Option<f64>,
        wall_landing: Option<WallLandingTarget>,
    ) -> Vec3 {
        let direction = planar_direction_override
            .map(|[x, y]| [x, y, 0.0])
            .unwrap_or_else(|| {
                let rotation = world.data().xmat()[self.root_body_id];
                [rotation[0], rotation[3], 0.0]
            });
        let direction_norm = direction[0]
            .hypot(direction[1])
            .hypot(direction[2])
            .max(1e-9);
        let mut target_velocity: [f64; 3] = std::array::from_fn(|axis| {
            horizontal_speed_scale * self.dynamics.target_horizontal_speed_mm_s * direction[axis]
                / direction_norm
        });
        if let Some(target) = wall_landing {
            let clearance = WALL_LANDING_FOOT_CLEARANCE_MM
                + WALL_LANDING_ROTATION_CLEARANCE_MM
                    * (1.0 - target.alignment(world.root_quaternion())).clamp(0.0, 1.0);
            let position = world.root_position();
            for axis in 0..2 {
                target_velocity[axis] = (WALL_LANDING_APPROACH_GAIN_PER_S
                    * (target.surface_point_mm[axis] + target.inward_xy[axis] * clearance
                        - position[axis]))
                    .clamp(
                        -WALL_LANDING_APPROACH_SPEED_MM_S,
                        WALL_LANDING_APPROACH_SPEED_MM_S,
                    );
            }
        }
        let velocity = world.root_velocity();
        let velocity_error_mm_s = [
            target_velocity[0] - velocity[3],
            target_velocity[1] - velocity[4],
        ];
        let maximum_vertical_force = self.body_mass_g
            * 9_810.0
            * if wall_landing.is_some() {
                2.0
            } else {
                self.dynamics.maximum_vertical_force_weight
            };
        let vertical_force = if let Some(target_height) = altitude_target_mm {
            let altitude_error_mm = target_height - world.root_position()[2];
            let altitude_base_force = self.body_mass_g
                * (self.dynamics.altitude_position_gain_per_s2 * altitude_error_mm
                    - self.dynamics.altitude_rate_gain_per_s * velocity[5]
                    + if wall_landing.is_some() { 9_810.0 } else { 0.0 });
            let integral_gain_per_s3 = self.dynamics.altitude_position_gain_per_s2
                / ENGINEERED_BODY_ALTITUDE_INTEGRAL_TIME_S;
            self.update_altitude_position_error_integral(
                world.time(),
                altitude_error_mm,
                altitude_base_force,
                maximum_vertical_force,
                integral_gain_per_s3,
            );
            (altitude_base_force
                + self.body_mass_g
                    * integral_gain_per_s3
                    * self.altitude_position_error_integral_mm_s)
                .clamp(-maximum_vertical_force, maximum_vertical_force)
        } else {
            self.reset_altitude_position_error_integral();
            0.0
        };
        let maximum_force =
            self.body_mass_g * 9_810.0 * self.dynamics.maximum_horizontal_force_weight;
        let integral_gain_per_s2 =
            self.dynamics.velocity_gain_per_s / ENGINEERED_BODY_VELOCITY_INTEGRAL_TIME_S;
        let proportional_force = [
            self.body_mass_g * self.dynamics.velocity_gain_per_s * velocity_error_mm_s[0],
            self.body_mass_g * self.dynamics.velocity_gain_per_s * velocity_error_mm_s[1],
        ];
        self.update_horizontal_velocity_integral(
            world.time(),
            velocity_error_mm_s,
            proportional_force,
            maximum_force,
            integral_gain_per_s2,
        );
        let integral_force_scale = self.body_mass_g * integral_gain_per_s2;
        let mut force = Vec3::new(
            proportional_force[0] + integral_force_scale * self.horizontal_velocity_integral_mm[0],
            proportional_force[1] + integral_force_scale * self.horizontal_velocity_integral_mm[1],
            vertical_force,
        );
        let magnitude = force.x().hypot(force.y());
        if magnitude > maximum_force {
            let scale = maximum_force / magnitude;
            force = Vec3::new(force.x() * scale, force.y() * scale, force.z());
        }
        force
    }

    fn reset_horizontal_velocity_integral(&mut self) {
        self.horizontal_velocity_integral_mm = [0.0; 2];
        self.last_horizontal_integral_time_s = None;
    }

    fn reset_altitude_position_error_integral(&mut self) {
        self.altitude_position_error_integral_mm_s = 0.0;
        self.last_altitude_integral_time_s = None;
    }

    fn update_altitude_position_error_integral(
        &mut self,
        time_seconds: f64,
        altitude_error_mm: f64,
        base_force_g_mm_s2: f64,
        maximum_force_g_mm_s2: f64,
        integral_gain_per_s3: f64,
    ) {
        let Some(last_time_seconds) = self.last_altitude_integral_time_s else {
            self.last_altitude_integral_time_s = Some(time_seconds);
            return;
        };
        let elapsed_seconds = time_seconds - last_time_seconds;
        if elapsed_seconds <= 0.0 {
            if elapsed_seconds < 0.0 {
                self.reset_altitude_position_error_integral();
                self.last_altitude_integral_time_s = Some(time_seconds);
            }
            return;
        }
        self.last_altitude_integral_time_s = Some(time_seconds);

        let integral_force_scale = self.body_mass_g * integral_gain_per_s3;
        let mut candidate_integral_mm =
            self.altitude_position_error_integral_mm_s + altitude_error_mm * elapsed_seconds;
        let candidate_integral_force = integral_force_scale * candidate_integral_mm;
        if candidate_integral_force.abs() > maximum_force_g_mm_s2 {
            candidate_integral_mm =
                candidate_integral_force.signum() * maximum_force_g_mm_s2 / integral_force_scale;
        }

        let current_force =
            base_force_g_mm_s2 + integral_force_scale * self.altitude_position_error_integral_mm_s;
        let candidate_force = base_force_g_mm_s2 + integral_force_scale * candidate_integral_mm;
        let current_force_is_saturated =
            current_force.abs() >= maximum_force_g_mm_s2 && altitude_error_mm * current_force > 0.0;
        let candidate_would_saturate_outward = candidate_force.abs() > maximum_force_g_mm_s2
            && altitude_error_mm * candidate_force > 0.0;
        if !current_force_is_saturated && !candidate_would_saturate_outward {
            self.altitude_position_error_integral_mm_s = candidate_integral_mm;
        }
    }

    fn update_horizontal_velocity_integral(
        &mut self,
        time_seconds: f64,
        velocity_error_mm_s: [f64; 2],
        proportional_force_g_mm_s2: [f64; 2],
        maximum_force_g_mm_s2: f64,
        integral_gain_per_s2: f64,
    ) {
        let Some(last_time_seconds) = self.last_horizontal_integral_time_s else {
            self.last_horizontal_integral_time_s = Some(time_seconds);
            return;
        };
        let elapsed_seconds = time_seconds - last_time_seconds;
        if elapsed_seconds <= 0.0 {
            if elapsed_seconds < 0.0 {
                self.reset_horizontal_velocity_integral();
                self.last_horizontal_integral_time_s = Some(time_seconds);
            }
            return;
        }
        self.last_horizontal_integral_time_s = Some(time_seconds);

        let mut candidate_integral_mm = [
            self.horizontal_velocity_integral_mm[0] + velocity_error_mm_s[0] * elapsed_seconds,
            self.horizontal_velocity_integral_mm[1] + velocity_error_mm_s[1] * elapsed_seconds,
        ];
        let integral_force_scale = self.body_mass_g * integral_gain_per_s2;
        let candidate_integral_force_norm =
            integral_force_scale * candidate_integral_mm[0].hypot(candidate_integral_mm[1]);
        if candidate_integral_force_norm > maximum_force_g_mm_s2 {
            let scale = maximum_force_g_mm_s2 / candidate_integral_force_norm;
            candidate_integral_mm[0] *= scale;
            candidate_integral_mm[1] *= scale;
        }

        let current_integral_force = [
            integral_force_scale * self.horizontal_velocity_integral_mm[0],
            integral_force_scale * self.horizontal_velocity_integral_mm[1],
        ];
        let candidate_integral_force = [
            integral_force_scale * candidate_integral_mm[0],
            integral_force_scale * candidate_integral_mm[1],
        ];
        let current_force = [
            proportional_force_g_mm_s2[0] + current_integral_force[0],
            proportional_force_g_mm_s2[1] + current_integral_force[1],
        ];
        let candidate_force = [
            proportional_force_g_mm_s2[0] + candidate_integral_force[0],
            proportional_force_g_mm_s2[1] + candidate_integral_force[1],
        ];
        let candidate_force_norm = candidate_force[0].hypot(candidate_force[1]);
        let error_force = [
            self.body_mass_g * self.dynamics.velocity_gain_per_s * velocity_error_mm_s[0],
            self.body_mass_g * self.dynamics.velocity_gain_per_s * velocity_error_mm_s[1],
        ];
        let current_force_is_saturated = current_force[0].hypot(current_force[1])
            >= maximum_force_g_mm_s2
            && error_force[0] * current_force[0] + error_force[1] * current_force[1] > 0.0;
        let candidate_would_saturate_outward = candidate_force_norm > maximum_force_g_mm_s2
            && error_force[0] * candidate_force[0] + error_force[1] * candidate_force[1] > 0.0;
        if !current_force_is_saturated && !candidate_would_saturate_outward {
            self.horizontal_velocity_integral_mm = candidate_integral_mm;
        }
    }

    fn wing_force(
        &self,
        world: &MuJoCoWorld,
        side_index: usize,
        air_velocity_mm_s: Vec3,
    ) -> Result<WingFlightTelemetry> {
        let wing = &self.config.wings[self.wing_indices[side_index]];
        let body_id = self.body_ids[side_index];
        let origin = Vec3::from(world.data().xpos()[body_id]);
        let center_of_mass = Vec3::from(world.data().xipos()[body_id]);
        let rotation = world.data().xmat()[body_id];
        let velocity = world
            .data()
            .try_object_velocity(MjtObj::mjOBJ_BODY, body_id, false)
            .context("reading aerodynamic wing velocity")?;
        let angular_velocity = Vec3::new(velocity[0], velocity[1], velocity[2]);
        let linear_velocity = Vec3::new(velocity[3], velocity[4], velocity[5]);
        let mut force = Vec3::ZERO;
        let mut moment = Vec3::ZERO;
        let mut peak_speed = 0.0_f64;
        let mut alpha_sum = 0.0;
        for strip in &wing.strips {
            let offset = rotate(rotation, strip.center_local_mm);
            let strip_position = origin + offset;
            let strip_velocity = linear_velocity + angular_velocity.cross(offset);
            let world_strip = transformed_strip(strip, rotation);
            let strip_force = self
                .config
                .strip_force(&world_strip, strip_velocity - air_velocity_mm_s)?;
            force = force + strip_force.force_g_mm_s2;
            moment = moment + (strip_position - center_of_mass).cross(strip_force.force_g_mm_s2);
            peak_speed = peak_speed.max(strip_force.speed_mm_s);
            alpha_sum += strip_force.angle_of_attack_rad;
        }
        Ok(WingFlightTelemetry {
            side: wing.side.clone(),
            force_g_mm_s2: force,
            moment_g_mm2_s2: moment,
            peak_strip_speed_mm_s: peak_speed,
            mean_angle_of_attack_rad: alpha_sum / wing.strips.len() as f64,
        })
    }

    fn zero_wing(&self, side_index: usize) -> WingFlightTelemetry {
        WingFlightTelemetry {
            side: self.config.wings[self.wing_indices[side_index]]
                .side
                .clone(),
            force_g_mm_s2: Vec3::ZERO,
            moment_g_mm2_s2: Vec3::ZERO,
            peak_strip_speed_mm_s: 0.0,
            mean_angle_of_attack_rad: 0.0,
        }
    }
}

fn wing_index(config: &AerodynamicsConfig, side: &str) -> Result<usize> {
    config
        .wings
        .iter()
        .position(|wing| wing.side == side)
        .with_context(|| format!("aerodynamics config is missing {side} wing"))
}

fn aerodynamic_body_id(
    config: &AerodynamicsConfig,
    world: &MuJoCoWorld,
    wing_index: usize,
) -> Result<usize> {
    let wing = &config.wings[wing_index];
    world
        .model()
        .name_to_id(MjtObj::mjOBJ_BODY, &wing.body)
        .with_context(|| format!("model is missing aerodynamic body {}", wing.body))
}

fn root_qfrc_fluid(world: &MuJoCoWorld) -> Result<[f64; 6]> {
    world
        .data()
        .qfrc_fluid()
        .get(..6)
        .and_then(|values| values.try_into().ok())
        .with_context(|| "MuJoCo qfrc_fluid has no free-root six-DOF slice")
}

fn compute_engineered_body_stabilizer_torque(
    world: &MuJoCoWorld,
    command: FlightCommand,
    dynamics: FlightDynamicsParameters,
) -> Vec3 {
    if let Some(target) = command.wall_landing {
        let [w, x, y, z] = world.root_quaternion();
        let [tw, tx, ty, tz] = target.orientation();
        let ew = tw * w + tx * x + ty * y + tz * z;
        let error = Vec3::new(
            -tw * x + tx * w - ty * z + tz * y,
            -tw * y + tx * z + ty * w - tz * x,
            -tw * z - tx * y + ty * x + tz * w,
        );
        let norm = error.norm();
        let rotation_error = error
            * (2.0 * norm.atan2(ew.abs()) * if ew < 0.0 { -1.0 } else { 1.0 } / norm.max(1e-12));
        let velocity = world.root_velocity();
        let pitch_axis = Vec3::new(
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        );
        let torque = rotation_error * dynamics.attitude_position_gain_g_mm2_s2
            - Vec3::new(velocity[0], velocity[1], velocity[2]) * dynamics.yaw_rate_gain_g_mm2_s2
            + pitch_axis
                * (dynamics.pitch_feedforward_torque_g_mm2_s2
                    * (command.frequency_scale * command.amplitude).powi(2));
        return torque
            * (dynamics.maximum_torque_g_mm2_s2
                / torque.norm().max(dynamics.maximum_torque_g_mm2_s2));
    }
    let [roll, pitch, yaw] = root_euler_rad(world.root_quaternion());
    let angular_velocity = world.root_velocity();
    let roll_rate = yaw.cos() * angular_velocity[0] + yaw.sin() * angular_velocity[1];
    let pitch_rate = -yaw.sin() * angular_velocity[0] + yaw.cos() * angular_velocity[1];
    let pitch_feedforward = dynamics.pitch_feedforward_torque_g_mm2_s2
        * (command.frequency_scale * command.amplitude).powi(2);
    let target_yaw_rate = command.heading_target_xy.map_or(
        dynamics.maximum_yaw_rate_rad_s * command.steering,
        |[x, y]| {
            let error = y.atan2(x) - yaw;
            (ENGINEERED_BODY_HEADING_GAIN_PER_S * error.sin().atan2(error.cos())).clamp(
                -dynamics.maximum_yaw_rate_rad_s,
                dynamics.maximum_yaw_rate_rad_s,
            )
        },
    );
    let roll_torque = (-dynamics.attitude_position_gain_g_mm2_s2 * roll
        - dynamics.attitude_rate_gain_g_mm2_s2 * roll_rate)
        .clamp(
            -dynamics.maximum_torque_g_mm2_s2,
            dynamics.maximum_torque_g_mm2_s2,
        );
    let pitch_torque = (pitch_feedforward
        - dynamics.attitude_position_gain_g_mm2_s2
            * (pitch
                - command
                    .body_pitch_target_rad
                    .unwrap_or(dynamics.target_pitch_rad))
        - dynamics.attitude_rate_gain_g_mm2_s2 * pitch_rate)
        .clamp(
            -dynamics.maximum_torque_g_mm2_s2,
            dynamics.maximum_torque_g_mm2_s2,
        );
    Vec3::new(
        yaw.cos() * roll_torque - yaw.sin() * pitch_torque,
        yaw.sin() * roll_torque + yaw.cos() * pitch_torque,
        (dynamics.yaw_rate_gain_g_mm2_s2 * (target_yaw_rate - angular_velocity[2])).clamp(
            -dynamics.maximum_torque_g_mm2_s2,
            dynamics.maximum_torque_g_mm2_s2,
        ),
    )
}

fn root_euler_rad(quaternion: [f64; 4]) -> [f64; 3] {
    let [w, x, y, z] = quaternion;
    [
        (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y)),
        (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin(),
        (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z)),
    ]
}

fn validate_mujoco_fluid_geoms(config: &AerodynamicsConfig, world: &MuJoCoWorld) -> Result<()> {
    let fluidcoef = config
        .model
        .fluidcoef
        .context("MuJoCo ellipsoid backend is missing fluidcoef")?;
    if (world.model().opt().density - config.air.rho_g_per_mm3).abs() > 1e-15
        || (world.model().opt().viscosity - config.air.dynamic_viscosity_g_per_mm_s).abs() > 1e-14
    {
        bail!("MuJoCo air density or viscosity does not match aerodynamics.json")
    }
    for name in &config.model.fluid_geom_names {
        let geom_id = world
            .model()
            .name_to_id(MjtObj::mjOBJ_GEOM, name)
            .with_context(|| format!("model is missing MuJoCo fluid geom {name}"))?;
        let fluid = world.model().geom_fluid()[geom_id];
        if fluid[0] <= 0.0
            || fluid[1..6]
                .iter()
                .zip(fluidcoef)
                .any(|(actual, expected)| (actual - expected).abs() > 1e-12)
        {
            bail!("MuJoCo fluid geom {name} has unexpected fluid coefficients")
        }
    }
    Ok(())
}

fn rotate(matrix: [f64; 9], vector: Vec3) -> Vec3 {
    Vec3::new(
        matrix[0] * vector.x() + matrix[1] * vector.y() + matrix[2] * vector.z(),
        matrix[3] * vector.x() + matrix[4] * vector.y() + matrix[5] * vector.z(),
        matrix[6] * vector.x() + matrix[7] * vector.y() + matrix[8] * vector.z(),
    )
}

fn transformed_strip(strip: &WingStrip, rotation: [f64; 9]) -> WingStrip {
    WingStrip {
        index: strip.index,
        center_local_mm: rotate(rotation, strip.center_local_mm),
        centroid_local_mm: strip
            .centroid_local_mm
            .map(|centroid| rotate(rotation, centroid)),
        span_hat_local: rotate(rotation, strip.span_hat_local),
        chord_hat_local: rotate(rotation, strip.chord_hat_local),
        normal_hat_local: rotate(rotation, strip.normal_hat_local),
        chord_mm: strip.chord_mm,
        width_mm: strip.width_mm,
        area_mm2: strip.area_mm2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wall_targets_point_feet_into_each_wall_without_euler_singularity() {
        for inward_xy in [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]] {
            let target = WallLandingTarget {
                surface_point_mm: [0.0, 0.0, 100.0],
                inward_xy,
            };
            let [w, x, y, z] = target.orientation();
            assert!((w * w + x * x + y * y + z * z - 1.0).abs() < 1e-12);
            assert!((target.alignment([w, x, y, z]) - 1.0).abs() < 1e-12);
            assert!((2.0 * (x * z - w * y) - 1.0).abs() < 1e-12);
            let command = FlightStabilizer::default()
                .command_with_base(
                    [w, x, y, z],
                    [0.0; 6],
                    FlightCommand {
                        enabled: true,
                        amplitude: 1.0,
                        wall_landing: Some(target),
                        ..Default::default()
                    },
                    0.8,
                )
                .unwrap();
            assert_eq!(command.pitch_bias_rad, 0.0);
            assert_eq!(command.differential_pitch_rad, 0.0);
        }
        assert!(
            FlightCommand {
                wall_landing: Some(WallLandingTarget {
                    surface_point_mm: [0.0; 3],
                    inward_xy: [0.0; 2],
                }),
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn wing_retraction_is_rate_limited_and_reset_safe() {
        let world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        let airborne = flight
            .controls(
                1.0,
                FlightCommand {
                    enabled: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let first = flight.controls(1.0001, FlightCommand::default()).unwrap();
        for (before, after) in airborne.into_iter().zip(first) {
            assert!((after - before).abs() <= 0.0100001);
        }
        assert_eq!(
            flight.controls(1.0001, FlightCommand::default()).unwrap(),
            first
        );
        assert_eq!(
            flight.controls(1.1, FlightCommand::default()).unwrap(),
            RETRACTED_WING_CONTROLS_RAD
        );
        assert_eq!(
            flight.controls(0.0, FlightCommand::default()).unwrap(),
            RETRACTED_WING_CONTROLS_RAD
        );
    }

    #[test]
    fn runtime_resolves_pinned_wings_and_has_no_force_when_disabled() {
        let mut world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        let telemetry = flight
            .apply(&mut world, FlightCommand::default(), [0.0; 3])
            .unwrap();
        assert_eq!(telemetry.total_force_g_mm_s2, Vec3::ZERO);
        assert_eq!(telemetry.root_qfrc_fluid, [0.0; 6]);
        assert_eq!(&world.controls()[50..56], &RETRACTED_WING_CONTROLS_RAD);
        assert!(!telemetry.engineered_body_stabilizer_enabled);
        assert_eq!(telemetry.engineered_body_velocity_force_g_mm_s2, Vec3::ZERO);
        assert_eq!(
            telemetry.engineered_body_stabilizer_torque_g_mm2_s2,
            Vec3::ZERO
        );
        assert!(telemetry.weight_g_mm_s2 > 0.0);
    }

    #[test]
    fn mujoco_ellipsoid_backend_has_massless_fluid_geoms() {
        let mut world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        assert!(flight.config().uses_mujoco_ellipsoid());
        assert_eq!(flight.config().model.fluid_geom_names.len(), 2);
        assert!((world.model().opt().density - 1.28e-6).abs() < 1e-15);
        assert!((world.model().opt().viscosity - 1.85e-5).abs() < 1e-14);
        for name in &flight.config().model.fluid_geom_names {
            let geom_id = world.model().name_to_id(MjtObj::mjOBJ_GEOM, name).unwrap();
            let fluid = world.model().geom_fluid()[geom_id];
            assert_eq!(&fluid[..6], &[1.0, 1.0, 0.5, 1.5, 1.7, 1.0]);
            assert!(
                world.model().geom_size()[geom_id]
                    .iter()
                    .all(|value| *value > 0.0)
            );
        }
        let command = FlightCommand {
            enabled: true,
            amplitude: 0.0,
            ..FlightCommand::default()
        };
        let telemetry = flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert_eq!(telemetry.root_qfrc_fluid, [0.0; 6]);
        let wing_body_ids = [
            world
                .model()
                .name_to_id(MjtObj::mjOBJ_BODY, "fly/l_wing")
                .unwrap(),
            world
                .model()
                .name_to_id(MjtObj::mjOBJ_BODY, "fly/r_wing")
                .unwrap(),
        ];
        assert!(
            wing_body_ids
                .iter()
                .all(|body_id| world.data().xfrc_applied()[*body_id] == [0.0; 6])
        );
    }

    #[test]
    fn runtime_sets_bounded_symmetric_wing_commands() {
        let mut world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        let telemetry = flight
            .apply(
                &mut world,
                FlightCommand {
                    enabled: true,
                    amplitude: 0.8,
                    steering: 0.0,
                    wing_steering_scale: 1.0,
                    horizontal_speed_scale: 1.0,
                    heading_target_xy: None,
                    planar_velocity_direction: None,
                    altitude_target_mm: None,
                    body_pitch_target_rad: None,
                    wall_landing: None,
                    frequency_scale: 1.0,
                    pitch_bias_rad: 0.0,
                    roll_bias_rad: 0.0,
                    differential_pitch_rad: 0.0,
                    differential_roll_rad: 0.0,
                },
                [0.0; 3],
            )
            .unwrap();
        assert_eq!(telemetry.controls_rad, world.controls()[50..56]);
        assert!(telemetry.controls_rad.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn wall_heading_control_can_suppress_wing_steering_without_changing_normal_steering() {
        let world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        let neutral = flight
            .controls(
                0.001,
                FlightCommand {
                    enabled: true,
                    amplitude: 0.8,
                    steering: 0.0,
                    ..FlightCommand::default()
                },
            )
            .unwrap();
        let suppressed = flight
            .controls(
                0.001,
                FlightCommand {
                    enabled: true,
                    amplitude: 0.8,
                    steering: 1.0,
                    wing_steering_scale: 0.0,
                    ..FlightCommand::default()
                },
            )
            .unwrap();
        let neural = flight
            .controls(
                0.001,
                FlightCommand {
                    enabled: true,
                    amplitude: 0.8,
                    steering: 1.0,
                    ..FlightCommand::default()
                },
            )
            .unwrap();
        assert_eq!(suppressed, neutral);
        assert_ne!(neural, neutral);
    }

    #[test]
    fn advance_matches_apply_then_step() {
        let mut expected_world = MuJoCoWorld::new().unwrap();
        let mut actual_world = MuJoCoWorld::new().unwrap();
        let mut expected_flight =
            FlightRuntime::new("assets/neuromechfly", &expected_world).unwrap();
        let mut actual_flight = FlightRuntime::new("assets/neuromechfly", &actual_world).unwrap();
        let command = FlightCommand {
            enabled: true,
            amplitude: 0.9,
            steering: 0.3,
            altitude_target_mm: Some(30.0),
            frequency_scale: 1.1,
            ..FlightCommand::default()
        };
        for world in [&mut expected_world, &mut actual_world] {
            world.data_mut().qpos_mut()[2] = 30.0;
            world.data_mut().qvel_mut()[0] = 150.0;
            world.data_mut().forward();
        }

        for _ in 0..1_000 {
            let expected = expected_flight
                .apply(&mut expected_world, command, [0.0; 3])
                .unwrap();
            expected_world.step().unwrap();
            let actual = actual_flight
                .advance(&mut actual_world, command, [0.0; 3])
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(actual_world.qpos(), expected_world.qpos());
            assert_eq!(actual_world.qvel(), expected_world.qvel());
            assert_eq!(actual_world.time(), expected_world.time());
        }
    }

    #[test]
    fn engineered_velocity_tracks_the_headward_body_axis() {
        let mut world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        world.data_mut().qpos_mut()[3..7].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        world.data_mut().qvel_mut()[..6].fill(0.0);
        world.data_mut().forward();
        let telemetry = flight
            .apply(
                &mut world,
                FlightCommand {
                    enabled: true,
                    amplitude: 1.0,
                    ..FlightCommand::default()
                },
                [0.0; 3],
            )
            .unwrap();
        assert!(telemetry.engineered_body_velocity_force_g_mm_s2.x() > 0.0);
        assert!(telemetry.engineered_body_velocity_force_g_mm_s2.y().abs() < 1e-12);
    }

    #[test]
    fn horizontal_integral_updates_once_for_repeated_apply() {
        let mut world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        world.data_mut().qvel_mut()[0] = 10.0;
        world.data_mut().qvel_mut()[1] = -5.0;
        world.data_mut().forward();
        let command = FlightCommand {
            enabled: true,
            amplitude: 1.0,
            horizontal_speed_scale: 0.0,
            ..FlightCommand::default()
        };

        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert_eq!(flight.horizontal_velocity_integral_mm, [0.0; 2]);
        world.step().unwrap();
        world.data_mut().qvel_mut()[0] = 10.0;
        world.data_mut().qvel_mut()[1] = -5.0;
        world.data_mut().forward();
        let advanced = flight.apply(&mut world, command, [0.0; 3]).unwrap();
        let integral_after_advance = flight.horizontal_velocity_integral_mm;
        assert!(
            integral_after_advance[0] < 0.0,
            "integral={integral_after_advance:?} velocity={:?}",
            world.root_velocity()
        );

        let repeated = flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert_eq!(
            flight.horizontal_velocity_integral_mm,
            integral_after_advance
        );
        assert_eq!(
            repeated.engineered_body_velocity_force_g_mm_s2,
            advanced.engineered_body_velocity_force_g_mm_s2
        );
    }

    #[test]
    fn altitude_integral_updates_once_for_repeated_apply() {
        let mut world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        let command = FlightCommand {
            enabled: true,
            amplitude: 1.0,
            horizontal_speed_scale: 0.0,
            altitude_target_mm: Some(30.0),
            ..FlightCommand::default()
        };
        world.data_mut().qpos_mut()[2] = 20.0;
        world.data_mut().qvel_mut()[5] = 0.0;
        world.data_mut().forward();

        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert_eq!(flight.altitude_position_error_integral_mm_s, 0.0);
        world.step().unwrap();
        world.data_mut().qpos_mut()[2] = 20.0;
        world.data_mut().qvel_mut()[5] = 0.0;
        world.data_mut().forward();
        let advanced = flight.apply(&mut world, command, [0.0; 3]).unwrap();
        let integral_after_advance = flight.altitude_position_error_integral_mm_s;
        assert!(integral_after_advance > 0.0);

        let repeated = flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert_eq!(
            flight.altitude_position_error_integral_mm_s,
            integral_after_advance
        );
        assert_eq!(
            repeated.engineered_body_velocity_force_g_mm_s2,
            advanced.engineered_body_velocity_force_g_mm_s2
        );
    }

    #[test]
    fn altitude_integral_resets_when_disabled_missing_target_or_time_rewinds() {
        let mut world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        let command = FlightCommand {
            enabled: true,
            amplitude: 1.0,
            horizontal_speed_scale: 0.0,
            altitude_target_mm: Some(30.0),
            ..FlightCommand::default()
        };
        world.data_mut().qpos_mut()[2] = 20.0;
        world.data_mut().qvel_mut()[5] = 0.0;
        world.data_mut().forward();

        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        world.step().unwrap();
        world.data_mut().qpos_mut()[2] = 20.0;
        world.data_mut().qvel_mut()[5] = 0.0;
        world.data_mut().forward();
        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert_ne!(flight.altitude_position_error_integral_mm_s, 0.0);

        flight
            .apply(
                &mut world,
                FlightCommand {
                    altitude_target_mm: None,
                    ..command
                },
                [0.0; 3],
            )
            .unwrap();
        assert_eq!(flight.altitude_position_error_integral_mm_s, 0.0);
        assert_eq!(flight.last_altitude_integral_time_s, None);

        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        world.step().unwrap();
        world.data_mut().qpos_mut()[2] = 20.0;
        world.data_mut().qvel_mut()[5] = 0.0;
        world.data_mut().forward();
        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert_ne!(flight.altitude_position_error_integral_mm_s, 0.0);

        flight
            .apply(&mut world, FlightCommand::default(), [0.0; 3])
            .unwrap();
        assert_eq!(flight.altitude_position_error_integral_mm_s, 0.0);
        assert_eq!(flight.last_altitude_integral_time_s, None);

        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        world.step().unwrap();
        world.data_mut().qpos_mut()[2] = 20.0;
        world.data_mut().qvel_mut()[5] = 0.0;
        world.data_mut().forward();
        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert_ne!(flight.altitude_position_error_integral_mm_s, 0.0);

        world.reset().unwrap();
        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert_eq!(flight.altitude_position_error_integral_mm_s, 0.0);
        assert_eq!(flight.last_altitude_integral_time_s, Some(0.0));
    }

    #[test]
    fn altitude_pi_force_is_clamped_without_windup() {
        let mut world = MuJoCoWorld::new().unwrap();
        let dynamics = FlightDynamicsParameters::default();
        let mut flight =
            FlightRuntime::new_with_parameters("assets/neuromechfly", &world, dynamics).unwrap();
        let command = FlightCommand {
            enabled: true,
            amplitude: 1.0,
            horizontal_speed_scale: 0.0,
            altitude_target_mm: Some(1_000.0),
            ..FlightCommand::default()
        };
        let maximum_force = flight.body_mass_g * 9_810.0 * dynamics.maximum_vertical_force_weight;
        world.data_mut().qpos_mut()[2] = 0.0;
        world.data_mut().qvel_mut()[5] = 0.0;
        world.data_mut().forward();

        for _ in 0..10 {
            let telemetry = flight.apply(&mut world, command, [0.0; 3]).unwrap();
            assert!(
                (telemetry.engineered_body_velocity_force_g_mm_s2.z().abs() - maximum_force).abs()
                    < 1e-9
            );
            assert_eq!(flight.altitude_position_error_integral_mm_s, 0.0);
            world.step().unwrap();
            world.data_mut().qpos_mut()[2] = 0.0;
            world.data_mut().qvel_mut()[5] = 0.0;
            world.data_mut().forward();
        }
    }

    #[test]
    fn horizontal_integral_resets_when_disabled_or_time_rewinds() {
        let mut world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        world.data_mut().qvel_mut()[0] = 10.0;
        world.data_mut().forward();
        let command = FlightCommand {
            enabled: true,
            amplitude: 1.0,
            horizontal_speed_scale: 0.0,
            ..FlightCommand::default()
        };

        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        world.step().unwrap();
        world.data_mut().qvel_mut()[0] = 10.0;
        world.data_mut().forward();
        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert_ne!(flight.horizontal_velocity_integral_mm, [0.0; 2]);

        flight
            .apply(&mut world, FlightCommand::default(), [0.0; 3])
            .unwrap();
        assert_eq!(flight.horizontal_velocity_integral_mm, [0.0; 2]);
        assert_eq!(flight.last_horizontal_integral_time_s, None);

        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert_eq!(flight.horizontal_velocity_integral_mm, [0.0; 2]);
        world.step().unwrap();
        world.data_mut().qvel_mut()[0] = 10.0;
        world.data_mut().forward();
        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert_ne!(flight.horizontal_velocity_integral_mm, [0.0; 2]);

        world.reset().unwrap();
        flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert_eq!(flight.horizontal_velocity_integral_mm, [0.0; 2]);
    }

    #[test]
    fn horizontal_pi_force_is_radially_clamped_without_windup() {
        let mut world = MuJoCoWorld::new().unwrap();
        let dynamics = FlightDynamicsParameters::default();
        let mut flight =
            FlightRuntime::new_with_parameters("assets/neuromechfly", &world, dynamics).unwrap();
        world.data_mut().qvel_mut()[0] = -1_000.0;
        world.data_mut().qvel_mut()[1] = 600.0;
        world.data_mut().forward();
        let command = FlightCommand {
            enabled: true,
            amplitude: 1.0,
            horizontal_speed_scale: 0.0,
            ..FlightCommand::default()
        };
        let maximum_force = flight.body_mass_g * 9_810.0 * dynamics.maximum_horizontal_force_weight;

        let first = flight.apply(&mut world, command, [0.0; 3]).unwrap();
        let first_force = first.engineered_body_velocity_force_g_mm_s2;
        assert!(
            (first_force.x().hypot(first_force.y()) - maximum_force).abs() < 1e-9,
            "force={first_force:?} maximum={maximum_force} velocity={:?}",
            world.root_velocity()
        );
        assert_eq!(flight.horizontal_velocity_integral_mm, [0.0; 2]);

        world.step().unwrap();
        world.data_mut().qvel_mut()[0] = -1_000.0;
        world.data_mut().qvel_mut()[1] = 600.0;
        world.data_mut().forward();
        let second = flight.apply(&mut world, command, [0.0; 3]).unwrap();
        let second_force = second.engineered_body_velocity_force_g_mm_s2;
        assert!(second_force.x().hypot(second_force.y()) <= maximum_force + 1e-9);
        assert_eq!(flight.horizontal_velocity_integral_mm, [0.0; 2]);
    }

    #[test]
    fn wingbeat_phase_is_continuous_across_frequency_changes() {
        let world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        let command = FlightCommand {
            enabled: true,
            amplitude: 1.0,
            frequency_scale: 1.25,
            ..FlightCommand::default()
        };

        flight.controls(0.0, command).unwrap();
        assert!((flight.wingbeat_phase_time_s - 0.0).abs() < 1e-12);
        flight.controls(0.001, command).unwrap();
        assert!((flight.wingbeat_phase_time_s - 0.00125).abs() < 1e-12);
        flight.controls(0.002, command).unwrap();
        assert!((flight.wingbeat_phase_time_s - 0.0025).abs() < 1e-12);
        let phase_before_repeat = flight.wingbeat_phase_time_s;
        flight
            .controls(
                0.002,
                FlightCommand {
                    frequency_scale: 0.5,
                    ..command
                },
            )
            .unwrap();
        assert!((flight.wingbeat_phase_time_s - phase_before_repeat).abs() < 1e-12);
        flight
            .controls(
                0.003,
                FlightCommand {
                    frequency_scale: 0.5,
                    ..command
                },
            )
            .unwrap();
        assert!((flight.wingbeat_phase_time_s - 0.003).abs() < 1e-12);

        flight.controls(0.002, FlightCommand::default()).unwrap();
        assert_eq!(flight.wingbeat_phase_time_s, 0.0);
        flight.controls(0.5, command).unwrap();
        assert_eq!(flight.wingbeat_phase_time_s, 0.0);
    }

    #[test]
    fn zero_amplitude_force_is_zero() {
        let mut world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        let telemetry = flight
            .apply(
                &mut world,
                FlightCommand {
                    enabled: true,
                    amplitude: 0.0,
                    ..FlightCommand::default()
                },
                [0.0; 3],
            )
            .unwrap();
        assert_eq!(telemetry.total_force_g_mm_s2, Vec3::ZERO);
        assert_eq!(telemetry.total_moment_g_mm2_s2, Vec3::ZERO);
        assert_eq!(telemetry.root_qfrc_fluid, [0.0; 6]);
    }

    #[test]
    fn engineered_body_surrogate_holds_target_pitch_with_bounded_torque() {
        let mut world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        let command = FlightCommand {
            enabled: true,
            amplitude: 1.0,
            ..FlightCommand::default()
        };
        let half_angle = ENGINEERED_FLIGHT_TARGET_PITCH_RAD * 0.5;
        world.data_mut().qpos_mut()[2] = 20.0;
        world.data_mut().qpos_mut()[3..7].copy_from_slice(&[
            half_angle.cos(),
            0.0,
            half_angle.sin(),
            0.0,
        ]);
        world.data_mut().qvel_mut()[..6].fill(0.0);
        world.data_mut().forward();
        let mut maximum_abs_roll = 0.0_f64;
        let mut maximum_abs_pitch_error = 0.0_f64;
        let mut maximum_abs_yaw = 0.0_f64;
        for _ in 0..1_000 {
            let telemetry = flight.apply(&mut world, command, [0.0; 3]).unwrap();
            assert!(telemetry.engineered_body_stabilizer_enabled);
            assert!(
                telemetry
                    .engineered_body_velocity_force_g_mm_s2
                    .x()
                    .hypot(telemetry.engineered_body_velocity_force_g_mm_s2.y())
                    <= telemetry.weight_g_mm_s2 * ENGINEERED_BODY_MAX_HORIZONTAL_FORCE_WEIGHT
                        + 1e-12
            );
            assert!(
                telemetry.engineered_body_velocity_force_g_mm_s2.z().abs()
                    <= telemetry.weight_g_mm_s2 * ENGINEERED_BODY_MAX_VERTICAL_FORCE_WEIGHT + 1e-12
            );
            assert!(
                telemetry
                    .engineered_body_stabilizer_torque_g_mm2_s2
                    .0
                    .iter()
                    .all(|value| value.abs() <= ENGINEERED_BODY_MAX_TORQUE_G_MM2_S2)
            );
            world.step().unwrap();
            let [roll, pitch, yaw] = root_euler_rad(world.root_quaternion());
            maximum_abs_roll = maximum_abs_roll.max(roll.abs());
            maximum_abs_pitch_error =
                maximum_abs_pitch_error.max((pitch - ENGINEERED_FLIGHT_TARGET_PITCH_RAD).abs());
            maximum_abs_yaw = maximum_abs_yaw.max(yaw.abs());
        }
        assert!(maximum_abs_roll < 0.25);
        assert!(maximum_abs_pitch_error < 0.25);
        assert!(maximum_abs_yaw < 0.25);
    }

    #[test]
    fn engineered_body_surrogate_is_stable_after_a_half_turn() {
        let mut world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        let command = FlightCommand {
            enabled: true,
            amplitude: 1.0,
            altitude_target_mm: Some(30.0),
            ..FlightCommand::default()
        };
        let half_yaw = std::f64::consts::FRAC_PI_2;
        let half_pitch = ENGINEERED_FLIGHT_TARGET_PITCH_RAD * 0.5;
        world.data_mut().qpos_mut()[..3].copy_from_slice(&[0.0, 0.0, 30.0]);
        world.data_mut().qpos_mut()[3..7].copy_from_slice(&[
            half_yaw.cos() * half_pitch.cos(),
            -half_yaw.sin() * half_pitch.sin(),
            half_yaw.cos() * half_pitch.sin(),
            half_yaw.sin() * half_pitch.cos(),
        ]);
        world.data_mut().qvel_mut()[..6].fill(0.0);
        world.data_mut().forward();

        let mut maximum_abs_roll = 0.0_f64;
        let mut maximum_abs_pitch_error = 0.0_f64;
        for _ in 0..1_000 {
            flight.advance(&mut world, command, [0.0; 3]).unwrap();
            let [roll, pitch, _] = root_euler_rad(world.root_quaternion());
            maximum_abs_roll = maximum_abs_roll.max(roll.abs());
            maximum_abs_pitch_error =
                maximum_abs_pitch_error.max((pitch - ENGINEERED_FLIGHT_TARGET_PITCH_RAD).abs());
        }
        let [w, x, y, z] = world.root_quaternion();
        let forward = [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z)];
        let heading_x = forward[0] / forward[0].hypot(forward[1]);
        assert!(maximum_abs_roll < 0.25);
        assert!(maximum_abs_pitch_error < 0.25);
        assert!(
            heading_x < -0.9,
            "forward={forward:?} quaternion={:?} velocity={:?}",
            world.root_quaternion(),
            world.root_velocity(),
        );
        assert!(world.root_position()[2] > 20.0);
    }

    #[test]
    fn engineered_altitude_tracker_moves_toward_a_higher_target() {
        let mut world = MuJoCoWorld::new().unwrap();
        let dynamics = FlightDynamicsParameters {
            target_horizontal_speed_mm_s: 1.0,
            ..FlightDynamicsParameters::default()
        };
        let mut flight =
            FlightRuntime::new_with_parameters("assets/neuromechfly", &world, dynamics).unwrap();
        let half_angle = dynamics.target_pitch_rad * 0.5;
        world.data_mut().qpos_mut()[0..3].copy_from_slice(&[0.0, 0.0, 28.0]);
        world.data_mut().qpos_mut()[3..7].copy_from_slice(&[
            half_angle.cos(),
            0.0,
            half_angle.sin(),
            0.0,
        ]);
        world.data_mut().qvel_mut()[..6].fill(0.0);
        world.data_mut().forward();
        let command = FlightCommand {
            enabled: true,
            amplitude: 1.0,
            altitude_target_mm: Some(60.0),
            frequency_scale: 1.3,
            ..FlightCommand::default()
        };
        for _ in 0..20_000 {
            flight.advance(&mut world, command, [0.0; 3]).unwrap();
        }
        let final_height = world.root_position()[2];
        assert!(final_height > 48.0, "final height {final_height}");
        assert!(final_height < 70.0, "final height {final_height}");
    }

    #[test]
    fn differential_controls_are_antisymmetric() {
        let mut world = MuJoCoWorld::new().unwrap();
        let mut flight = FlightRuntime::new("assets/neuromechfly", &world).unwrap();
        let command = FlightCommand {
            enabled: true,
            amplitude: 0.8,
            differential_pitch_rad: 0.1,
            differential_roll_rad: -0.08,
            ..FlightCommand::default()
        };
        let telemetry = flight.apply(&mut world, command, [0.0; 3]).unwrap();
        assert!((telemetry.controls_rad[1] - telemetry.controls_rad[4]).abs() > 0.19);
        assert!((telemetry.controls_rad[2] - telemetry.controls_rad[5]).abs() > 0.15);
    }

    #[test]
    fn projected_feedback_stays_inside_worst_case_ranges() {
        let config = AerodynamicsConfig::load("assets/neuromechfly").unwrap();
        let command = project_attitude_feedback(
            FlightCommand {
                enabled: true,
                amplitude: 1.0,
                pitch_bias_rad: 0.9,
                roll_bias_rad: -0.9,
                differential_pitch_rad: -0.9,
                differential_roll_rad: 0.9,
                ..FlightCommand::default()
            },
            &config,
        )
        .unwrap();
        for (axis, bias, differential) in [
            (
                "pitch",
                command.pitch_bias_rad,
                command.differential_pitch_rad,
            ),
            ("roll", command.roll_bias_rad, command.differential_roll_rad),
        ] {
            let center = config.wingbeat.center_rad_by_axis[axis];
            let amplitude = config.wingbeat.amplitude_rad_by_axis[axis];
            let range = config.wingbeat.joint_ranges_rad[axis];
            for sign in [-1.0, 1.0] {
                let control = center + sign * amplitude + bias + differential;
                let mirrored = center + sign * amplitude + bias - differential;
                assert!(control >= range[0] - 1e-12 && control <= range[1] + 1e-12);
                assert!(mirrored >= range[0] - 1e-12 && mirrored <= range[1] + 1e-12);
            }
        }
    }

    #[test]
    fn rotation_preserves_basis_vectors() {
        let rotation = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(
            rotate(rotation, Vec3::new(1.0, 0.0, 0.0)),
            Vec3::new(0.0, 1.0, 0.0)
        );
        assert_eq!(
            rotate(rotation, Vec3::new(0.0, 1.0, 0.0)),
            Vec3::new(-1.0, 0.0, 0.0)
        );
    }

    #[test]
    fn stabilizer_opposes_positive_pitch_and_roll() {
        let stabilizer = FlightStabilizer::default();
        let pitch = stabilizer
            .command([0.995004, 0.0, 0.0998334, 0.0], [0.0; 6], 1.0)
            .unwrap();
        let roll = stabilizer
            .command([0.995004, 0.0998334, 0.0, 0.0], [0.0; 6], 1.0)
            .unwrap();
        assert!(pitch.pitch_bias_rad < stabilizer.pitch_bias_rad);
        assert!(roll.differential_pitch_rad < 0.0);
    }

    #[test]
    fn engineered_yaw_rate_tracks_the_descending_steering_command() {
        let mut world = MuJoCoWorld::new().unwrap();
        let half_angle = ENGINEERED_FLIGHT_TARGET_PITCH_RAD * 0.5;
        world.data_mut().qpos_mut()[3..7].copy_from_slice(&[
            half_angle.cos(),
            0.0,
            half_angle.sin(),
            0.0,
        ]);
        world.data_mut().qvel_mut()[..6].fill(0.0);
        world.data_mut().forward();
        let positive = compute_engineered_body_stabilizer_torque(
            &world,
            FlightCommand {
                enabled: true,
                amplitude: 1.0,
                steering: 1.0,
                ..FlightCommand::default()
            },
            FlightDynamicsParameters::default(),
        );
        let negative = compute_engineered_body_stabilizer_torque(
            &world,
            FlightCommand {
                enabled: true,
                amplitude: 1.0,
                steering: -1.0,
                ..FlightCommand::default()
            },
            FlightDynamicsParameters::default(),
        );
        assert!(positive.z() > 0.0);
        assert!(negative.z() < 0.0);
        assert!((positive.z() + negative.z()).abs() < 1e-12);
    }
}
