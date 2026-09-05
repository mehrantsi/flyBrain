use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::flight::{FlightCommand, FlightDynamicsParameters, FlightRuntime, FlightStabilizer};
use crate::system_id::{
    MetricObservation, MetricTarget, NamedParameter, ParameterVector, TrialRecord, huber_loss,
};
use crate::world::MuJoCoWorld;

const TARGET_HEIGHT_MM: f64 = 30.0;
const COMMAND_AMPLITUDE: f64 = 0.94;
const COMMAND_FREQUENCY_SCALE: f64 = 1.25;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FlightManeuverPrediction {
    pub family: String,
    pub reflection: String,
    pub observations: Vec<MetricObservation>,
}

pub struct FlightCalibrationEvaluator {
    assets: PathBuf,
    duration_by_family: BTreeMap<String, f64>,
    initial_forward_speed_mm_s: f64,
}

impl FlightCalibrationEvaluator {
    pub fn from_training_trials(
        assets: impl AsRef<Path>,
        training_trials: &[TrialRecord],
    ) -> Result<Self> {
        if training_trials.is_empty() {
            bail!("flight calibration requires training trials")
        }
        let mut durations = BTreeMap::<String, Vec<f64>>::new();
        let mut speeds = Vec::new();
        for trial in training_trials {
            let family = condition(trial, "family")?;
            durations
                .entry(family.to_string())
                .or_default()
                .push(observation(trial, "duration_seconds")?);
            speeds.push(observation(trial, "forward_speed_mean_mm_s")?);
        }
        let duration_by_family = durations
            .into_iter()
            .map(|(family, values)| (family, mean(&values)))
            .collect();
        let initial_forward_speed_mm_s = mean(&speeds).max(1.0);
        Ok(Self {
            assets: assets.as_ref().to_path_buf(),
            duration_by_family,
            initial_forward_speed_mm_s,
        })
    }

    pub fn parameter_vector(parameters: FlightDynamicsParameters) -> Result<ParameterVector> {
        ParameterVector::new(vec![
            NamedParameter::new(
                "target_pitch_rad",
                -1.10,
                -0.55,
                parameters.target_pitch_rad,
            ),
            NamedParameter::new(
                "target_horizontal_speed_mm_s",
                60.0,
                360.0,
                parameters.target_horizontal_speed_mm_s.clamp(60.0, 360.0),
            ),
            NamedParameter::new(
                "maximum_yaw_rate_rad_s",
                1.0,
                20.0,
                parameters.maximum_yaw_rate_rad_s,
            ),
            NamedParameter::new(
                "attitude_position_gain_g_mm2_s2",
                10.0,
                120.0,
                parameters.attitude_position_gain_g_mm2_s2,
            ),
            NamedParameter::new(
                "attitude_rate_gain_g_mm2_s2",
                0.005,
                0.25,
                parameters.attitude_rate_gain_g_mm2_s2,
            ),
            NamedParameter::new(
                "yaw_rate_gain_g_mm2_s2",
                0.5,
                25.0,
                parameters.yaw_rate_gain_g_mm2_s2,
            ),
            NamedParameter::new(
                "maximum_torque_g_mm2_s2",
                15.0,
                140.0,
                parameters.maximum_torque_g_mm2_s2,
            ),
            NamedParameter::new(
                "velocity_gain_per_s",
                2.0,
                200.0,
                parameters.velocity_gain_per_s,
            ),
            NamedParameter::new(
                "maximum_horizontal_force_weight",
                0.05,
                2.0,
                parameters.maximum_horizontal_force_weight,
            ),
        ])
    }

    pub fn dynamics_from_vector(
        base: FlightDynamicsParameters,
        parameters: &ParameterVector,
    ) -> Result<FlightDynamicsParameters> {
        parameters.validate()?;
        let mut dynamics = base;
        dynamics.target_pitch_rad = required_parameter(parameters, "target_pitch_rad")?;
        dynamics.target_horizontal_speed_mm_s =
            required_parameter(parameters, "target_horizontal_speed_mm_s")?;
        dynamics.maximum_yaw_rate_rad_s = required_parameter(parameters, "maximum_yaw_rate_rad_s")?;
        dynamics.attitude_position_gain_g_mm2_s2 =
            required_parameter(parameters, "attitude_position_gain_g_mm2_s2")?;
        dynamics.attitude_rate_gain_g_mm2_s2 =
            required_parameter(parameters, "attitude_rate_gain_g_mm2_s2")?;
        dynamics.yaw_rate_gain_g_mm2_s2 = required_parameter(parameters, "yaw_rate_gain_g_mm2_s2")?;
        dynamics.maximum_torque_g_mm2_s2 =
            required_parameter(parameters, "maximum_torque_g_mm2_s2")?;
        dynamics.velocity_gain_per_s = required_parameter(parameters, "velocity_gain_per_s")?;
        dynamics.maximum_horizontal_force_weight =
            required_parameter(parameters, "maximum_horizontal_force_weight")?;
        dynamics.validate()
    }

    pub fn objective(
        &self,
        parameters: &ParameterVector,
        trials: &[TrialRecord],
        metric_specs: &[MetricTarget],
        base: FlightDynamicsParameters,
    ) -> Result<f64> {
        let dynamics = Self::dynamics_from_vector(base, parameters)?;
        let mut predictions = BTreeMap::new();
        let mut weighted_loss = 0.0;
        let mut total_weight = 0.0;
        for trial in trials {
            let family = condition(trial, "family")?;
            let reflection = condition(trial, "reflection")?;
            let key = (family.to_string(), reflection.to_string());
            if !predictions.contains_key(&key) {
                let prediction = self.simulate(dynamics, family, reflection)?;
                predictions.insert(key.clone(), prediction);
            }
            let prediction = &predictions[&key];
            for target in &trial.observations {
                let Some(spec) = metric_specs.iter().find(|spec| spec.name == target.name) else {
                    continue;
                };
                let Some(predicted) = prediction
                    .observations
                    .iter()
                    .find(|observed| observed.name == target.name)
                else {
                    continue;
                };
                let normalized = (predicted.value - target.value) / spec.scale;
                weighted_loss += spec.weight * huber_loss(normalized, spec.huber_delta)?;
                total_weight += spec.weight;
            }
        }
        if total_weight == 0.0 {
            bail!("flight calibration trials contain no supported weighted metrics")
        }
        Ok(weighted_loss / total_weight)
    }

    pub fn simulate(
        &self,
        dynamics: FlightDynamicsParameters,
        family: &str,
        reflection: &str,
    ) -> Result<FlightManeuverPrediction> {
        let duration_seconds = *self
            .duration_by_family
            .get(family)
            .with_context(|| format!("training data has no duration for flight family {family}"))?;
        let steering_magnitude = match family {
            "evasion" => 0.65,
            "saccade" => 1.0,
            _ => bail!("unsupported flight family {family}"),
        };
        let steering = match reflection {
            "original" => steering_magnitude,
            "reflected" => -steering_magnitude,
            _ => bail!("unsupported flight reflection {reflection}"),
        };
        simulate_maneuver(
            &self.assets,
            dynamics,
            duration_seconds,
            self.initial_forward_speed_mm_s,
            steering,
            family,
            reflection,
        )
    }
}

fn simulate_maneuver(
    assets: &Path,
    dynamics: FlightDynamicsParameters,
    duration_seconds: f64,
    initial_forward_speed_mm_s: f64,
    steering: f64,
    family: &str,
    reflection: &str,
) -> Result<FlightManeuverPrediction> {
    let mut world = MuJoCoWorld::from_assets_dir(assets)?;
    let mut flight = FlightRuntime::new_with_parameters(assets, &world, dynamics)?;
    let stabilizer = FlightStabilizer::from_dynamics(dynamics)?;
    let timestep = world.timestep_seconds();
    let steps = (duration_seconds / timestep).round() as usize;
    if steps == 0 {
        bail!("flight maneuver duration is shorter than one physics step")
    }
    world.data_mut().qpos_mut()[0..3].copy_from_slice(&[0.0, 0.0, TARGET_HEIGHT_MM]);
    let half_pitch = dynamics.target_pitch_rad * 0.5;
    world.data_mut().qpos_mut()[3..7].copy_from_slice(&[
        half_pitch.cos(),
        0.0,
        half_pitch.sin(),
        0.0,
    ]);
    world.data_mut().qvel_mut()[..6].fill(0.0);
    world.data_mut().qvel_mut()[0] = initial_forward_speed_mm_s;
    world.data_mut().forward();

    let command = FlightCommand {
        enabled: true,
        amplitude: COMMAND_AMPLITUDE,
        steering,
        wing_steering_scale: 1.0,
        horizontal_speed_scale: 1.0,
        heading_target_xy: None,
        planar_velocity_direction: None,
        altitude_target_mm: Some(TARGET_HEIGHT_MM),
        body_pitch_target_rad: None,
        wall_landing: None,
        frequency_scale: COMMAND_FREQUENCY_SCALE,
        pitch_bias_rad: 0.0,
        roll_bias_rad: 0.0,
        differential_pitch_rad: 0.0,
        differential_roll_rad: 0.0,
    };
    let initial_yaw = euler_rad(world.root_quaternion())[2];
    let mut previous_yaw = initial_yaw;
    let mut unwrapped_yaw = initial_yaw;
    let mut pitch_sum = 0.0;
    let mut planar_speed_sum = 0.0;
    let mut forward_speed_sum = 0.0;
    let mut root_yaw_sum = 0.0;
    let mut yaw_rate_sum = 0.0;
    let mut command_steering_sum = 0.0;
    let mut command_forward_speed_sum = 0.0;
    let mut absolute_turn_rate_sum = 0.0;
    let mut angular_speed_sum = 0.0;
    let mut vertical_speed_sum = 0.0;
    for step in 0..steps {
        let ramp = ((step as f64 * timestep) / 0.02).clamp(0.0, 1.0);
        let effective = stabilizer.command_with_base_limited(
            world.root_quaternion(),
            world.root_velocity(),
            command,
            ramp,
            flight.config(),
        )?;
        flight.advance(&mut world, effective, [0.0; 3])?;
        let [roll, pitch, yaw] = euler_rad(world.root_quaternion());
        let _ = roll;
        let yaw_delta = (yaw - previous_yaw).sin().atan2((yaw - previous_yaw).cos());
        unwrapped_yaw += yaw_delta;
        previous_yaw = yaw;
        let velocity = world.root_velocity();
        let forward = body_forward(world.root_quaternion());
        pitch_sum += pitch;
        planar_speed_sum += velocity[3].hypot(velocity[4]);
        forward_speed_sum += velocity[3] * forward[0] + velocity[4] * forward[1];
        root_yaw_sum += unwrapped_yaw;
        yaw_rate_sum += velocity[2];
        command_steering_sum += command.steering;
        command_forward_speed_sum +=
            command.horizontal_speed_scale * dynamics.target_horizontal_speed_mm_s;
        absolute_turn_rate_sum += velocity[2].abs();
        angular_speed_sum += velocity[0].hypot(velocity[1]).hypot(velocity[2]);
        vertical_speed_sum += velocity[5];
    }
    let count = steps as f64;
    Ok(FlightManeuverPrediction {
        family: family.to_string(),
        reflection: reflection.to_string(),
        observations: vec![
            MetricObservation::new("pitch_mean_rad", pitch_sum / count),
            MetricObservation::new("planar_speed_mean_mm_s", planar_speed_sum / count),
            MetricObservation::new("forward_speed_mean_mm_s", forward_speed_sum / count),
            MetricObservation::new("root_yaw_mean_rad", root_yaw_sum / count),
            MetricObservation::new("yaw_rate_mean_rad_s", yaw_rate_sum / count),
            MetricObservation::new("command_steering_mean", command_steering_sum / count),
            MetricObservation::new(
                "command_forward_speed_mean_mm_s",
                command_forward_speed_sum / count,
            ),
            MetricObservation::new(
                "absolute_turn_rate_mean_rad_s",
                absolute_turn_rate_sum / count,
            ),
            MetricObservation::new("angular_speed_mean_rad_s", angular_speed_sum / count),
            MetricObservation::new("vertical_speed_mean_mm_s", vertical_speed_sum / count),
            MetricObservation::new(
                "heading_rate_abs_rad_s",
                (unwrapped_yaw - initial_yaw).abs() / duration_seconds,
            ),
            MetricObservation::new("duration_seconds", duration_seconds),
        ],
    })
}

fn condition<'a>(trial: &'a TrialRecord, key: &str) -> Result<&'a str> {
    trial
        .condition
        .get(key)
        .map(String::as_str)
        .with_context(|| format!("trial {} is missing condition {key}", trial.trial_id))
}

fn observation(trial: &TrialRecord, name: &str) -> Result<f64> {
    trial
        .observations
        .iter()
        .find(|observation| observation.name == name)
        .map(|observation| observation.value)
        .with_context(|| format!("trial {} is missing observation {name}", trial.trial_id))
}

fn required_parameter(parameters: &ParameterVector, name: &str) -> Result<f64> {
    parameters
        .get(name)
        .with_context(|| format!("flight calibration parameter vector is missing {name}"))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn body_forward(quaternion: [f64; 4]) -> [f64; 2] {
    let [w, x, y, z] = quaternion;
    [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z)]
}

fn euler_rad(quaternion: [f64; 4]) -> [f64; 3] {
    let [w, x, y, z] = quaternion;
    [
        (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y)),
        (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin(),
        (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flight_parameter_vector_round_trips_defaults() {
        let defaults = FlightDynamicsParameters::default();
        let vector = FlightCalibrationEvaluator::parameter_vector(defaults).unwrap();
        let restored = FlightCalibrationEvaluator::dynamics_from_vector(defaults, &vector).unwrap();
        assert_eq!(restored, defaults);
    }
}
