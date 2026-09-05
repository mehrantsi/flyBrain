use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

pub const TAKEOFF_DRIVE_THRESHOLD: f64 = 0.015;
pub const TAKEOFF_DRIVE_DWELL_SECONDS: f64 = 0.15;
const SURFACE_CONTACT_DWELL_SECONDS: f64 = 0.03;
const NEURAL_ALTITUDE_ACTIVATION: f64 = 0.025;
const NEURAL_ALTITUDE_RELEASE: f64 = 0.02;
const NEURAL_ALTITUDE_MOTOR_GAIN: f64 = 4.0;
const NEURAL_ALTITUDE_MINIMUM_COMMAND: f64 = 0.35;
const NEURAL_ALTITUDE_BOUT_SECONDS: f64 = 2.0;
const NEURAL_FLIGHT_DRIVE_ALTITUDE_RANGE: f64 = 0.065;
const NEURAL_FLIGHT_DRIVE_MAXIMUM_HEIGHT_FRACTION: f64 = 0.75;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlightBehaviorParameters {
    pub wander_tau_seconds: f64,
    pub takeoff_drive_threshold: f64,
    pub takeoff_drive_dwell_seconds: f64,
    pub landing_odor_threshold: f64,
    pub landing_bilateral_fraction: f64,
    pub odor_steering_weight: f64,
    #[serde(default = "default_odor_gradient_gain")]
    pub odor_gradient_gain: f64,
    pub wander_steering_weight: f64,
    pub brain_steering_weight: f64,
    pub maximum_abs_steering: f64,
    pub takeoff_height_mm: f64,
    pub cruise_height_mm: f64,
    pub landing_height_mm: f64,
    pub takeoff_amplitude: f64,
    pub cruise_amplitude: f64,
    pub landing_amplitude: f64,
    pub brain_amplitude_gain: f64,
    #[serde(default = "default_neural_altitude_rate_mm_s")]
    pub neural_altitude_rate_mm_s: f64,
    #[serde(default = "default_optic_flow_altitude_rate_mm_s")]
    pub optic_flow_altitude_rate_mm_s: f64,
}

fn default_neural_altitude_rate_mm_s() -> f64 {
    40.0
}

fn default_odor_gradient_gain() -> f64 {
    100.0
}

fn default_optic_flow_altitude_rate_mm_s() -> f64 {
    24.0
}

impl Default for FlightBehaviorParameters {
    fn default() -> Self {
        Self {
            wander_tau_seconds: 0.7,
            takeoff_drive_threshold: TAKEOFF_DRIVE_THRESHOLD,
            takeoff_drive_dwell_seconds: TAKEOFF_DRIVE_DWELL_SECONDS,
            landing_odor_threshold: 1.1,
            landing_bilateral_fraction: 0.18,
            odor_steering_weight: 0.58,
            odor_gradient_gain: default_odor_gradient_gain(),
            wander_steering_weight: 0.16,
            brain_steering_weight: 0.26,
            maximum_abs_steering: 0.7,
            takeoff_height_mm: 18.0,
            cruise_height_mm: 28.0,
            landing_height_mm: 0.6,
            takeoff_amplitude: 1.0,
            cruise_amplitude: 0.94,
            landing_amplitude: 0.94,
            brain_amplitude_gain: 0.06,
            neural_altitude_rate_mm_s: default_neural_altitude_rate_mm_s(),
            optic_flow_altitude_rate_mm_s: 24.0,
        }
    }
}

impl FlightBehaviorParameters {
    pub fn validate(self) -> Result<Self> {
        let positive = [
            self.wander_tau_seconds,
            self.takeoff_drive_dwell_seconds,
            self.landing_odor_threshold,
            self.takeoff_height_mm,
            self.cruise_height_mm,
            self.landing_height_mm,
        ];
        if positive
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
            || !self.takeoff_drive_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.takeoff_drive_threshold)
            || !self.landing_bilateral_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.landing_bilateral_fraction)
            || [
                self.odor_steering_weight,
                self.odor_gradient_gain,
                self.wander_steering_weight,
                self.brain_steering_weight,
                self.maximum_abs_steering,
                self.takeoff_amplitude,
                self.cruise_amplitude,
                self.landing_amplitude,
                self.brain_amplitude_gain,
                self.neural_altitude_rate_mm_s,
                self.optic_flow_altitude_rate_mm_s,
            ]
            .into_iter()
            .any(|value| !value.is_finite() || value < 0.0)
            || self.maximum_abs_steering > 1.0
            || self.takeoff_amplitude > 1.0
            || self.cruise_amplitude > 1.0
            || self.landing_amplitude > 1.0
        {
            bail!("flight behavior parameters are invalid")
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlightMode {
    #[default]
    Grounded,
    Takeoff,
    Cruise,
    Landing,
}

impl FlightMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Grounded => "GROUNDED",
            Self::Takeoff => "TAKEOFF",
            Self::Cruise => "CRUISE",
            Self::Landing => "LANDING",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlightBehaviorInput {
    pub dt_seconds: f64,
    pub enabled: bool,
    pub brain_enabled: bool,
    pub root_height_mm: f64,
    pub vertical_velocity_mm_s: f64,
    pub angular_speed_rad_s: f64,
    pub contact_count: usize,
    pub odor_left: f64,
    pub odor_right: f64,
    pub taste_valence: f64,
    pub brain_flight_drive: f64,
    pub cns_motor_activation: Option<f64>,
    pub brain_steering: f64,
    pub brain_altitude_control: f64,
    pub optic_flow_altitude_control: f64,
    pub altitude_hold: bool,
    pub cns_food_approach: bool,
    pub cns_approach_height_mm: Option<f64>,
    pub landing_request: bool,
    pub takeoff_inhibited: bool,
    pub collision_escape_active: bool,
    pub flight_altitude_bounds_mm: [f64; 2],
}

impl Default for FlightBehaviorInput {
    fn default() -> Self {
        Self {
            dt_seconds: 0.0,
            enabled: false,
            brain_enabled: false,
            root_height_mm: 0.0,
            vertical_velocity_mm_s: 0.0,
            angular_speed_rad_s: 0.0,
            contact_count: 0,
            odor_left: 0.0,
            odor_right: 0.0,
            taste_valence: 0.0,
            brain_flight_drive: 0.0,
            cns_motor_activation: None,
            brain_steering: 0.0,
            brain_altitude_control: 0.0,
            optic_flow_altitude_control: 0.0,
            altitude_hold: false,
            cns_food_approach: false,
            cns_approach_height_mm: None,
            landing_request: false,
            takeoff_inhibited: false,
            collision_escape_active: false,
            flight_altitude_bounds_mm: [5.0, 208.0],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlightBehaviorCommand {
    pub mode: FlightMode,
    pub amplitude_scale: f64,
    pub steering: f64,
    pub target_height_mm: f64,
    pub altitude_target_clamped: bool,
    pub odor_steering_contribution: f64,
    pub wander_steering_contribution: f64,
    pub brain_steering_contribution: f64,
    pub neural_altitude_contribution_mm_s: f64,
    pub optic_flow_altitude_contribution_mm_s: f64,
}

pub struct FlightBehaviorController {
    parameters: FlightBehaviorParameters,
    mode: FlightMode,
    mode_elapsed_seconds: f64,
    takeoff_drive_elapsed_seconds: f64,
    surface_contact_elapsed_seconds: f64,
    landing_load_transfer: f64,
    landing_contact_lost_seconds: f64,
    random_state: u64,
    wander: f64,
    cruise_target_height_mm: f64,
    neural_altitude_input_active: bool,
    neural_altitude_motor_command: f64,
    neural_altitude_bout_remaining_seconds: f64,
    airborne_peak_flight_drive: f64,
    cns_motor_silence_seconds: f64,
}

impl FlightBehaviorController {
    pub fn new(seed: u64) -> Self {
        Self::with_parameters(seed, FlightBehaviorParameters::default())
            .expect("default flight behavior parameters are valid")
    }

    pub fn with_parameters(seed: u64, parameters: FlightBehaviorParameters) -> Result<Self> {
        Ok(Self {
            parameters: parameters.validate()?,
            mode: FlightMode::Grounded,
            mode_elapsed_seconds: 0.0,
            takeoff_drive_elapsed_seconds: 0.0,
            surface_contact_elapsed_seconds: 0.0,
            landing_load_transfer: 0.0,
            landing_contact_lost_seconds: 0.0,
            random_state: seed,
            wander: 0.0,
            cruise_target_height_mm: parameters.cruise_height_mm,
            neural_altitude_input_active: false,
            neural_altitude_motor_command: 0.0,
            neural_altitude_bout_remaining_seconds: 0.0,
            airborne_peak_flight_drive: 0.0,
            cns_motor_silence_seconds: 0.0,
        })
    }

    pub fn reset(&mut self, seed: u64) {
        *self = Self::with_parameters(seed, self.parameters)
            .expect("existing flight behavior parameters are valid");
    }

    pub fn parameters(&self) -> FlightBehaviorParameters {
        self.parameters
    }

    pub fn update(&mut self, input: FlightBehaviorInput) -> Result<FlightBehaviorCommand> {
        validate_input(input)?;
        if !input.enabled || input.cns_motor_activation == Some(0.0) {
            self.enter(FlightMode::Grounded);
            self.takeoff_drive_elapsed_seconds = 0.0;
            return Ok(FlightBehaviorCommand::default());
        }

        self.mode_elapsed_seconds += input.dt_seconds;
        if let Some(power) = input.cns_motor_activation {
            if power < self.parameters.takeoff_drive_threshold * 0.5 {
                self.cns_motor_silence_seconds += input.dt_seconds;
            } else {
                self.cns_motor_silence_seconds = 0.0;
            }
            if self.cns_motor_silence_seconds >= 0.5
                && matches!(self.mode, FlightMode::Takeoff | FlightMode::Cruise)
            {
                self.enter(FlightMode::Landing);
            }
        }
        let random = self.next_unit() * 2.0 - 1.0;
        let wander_alpha = 1.0 - (-input.dt_seconds / self.parameters.wander_tau_seconds).exp();
        self.wander += wander_alpha * (random - self.wander);

        let odor_total = input.odor_left + input.odor_right;
        if self.mode == FlightMode::Landing {
            self.landing_contact_lost_seconds = if input.contact_count >= 2 {
                0.0
            } else {
                self.landing_contact_lost_seconds + input.dt_seconds
            };
            let support_held = input.contact_count >= 2
                || (self.landing_load_transfer > 0.0 && self.landing_contact_lost_seconds <= 0.02);
            let direction = if support_held { 1.0 } else { -1.0 };
            self.landing_load_transfer =
                (self.landing_load_transfer + direction * input.dt_seconds / 0.08).clamp(0.0, 1.0);
        }
        if matches!(self.mode, FlightMode::Cruise | FlightMode::Landing) && input.contact_count >= 2
        {
            self.surface_contact_elapsed_seconds += input.dt_seconds;
        } else {
            self.surface_contact_elapsed_seconds = 0.0;
        }
        if self.mode == FlightMode::Grounded
            && !input.takeoff_inhibited
            && input.taste_valence <= 0.0
            && input.brain_flight_drive >= self.parameters.takeoff_drive_threshold
        {
            self.takeoff_drive_elapsed_seconds += input.dt_seconds;
        } else if self.mode == FlightMode::Grounded {
            self.takeoff_drive_elapsed_seconds = 0.0;
        }
        match self.mode {
            FlightMode::Grounded
                if !input.takeoff_inhibited
                    && input.taste_valence <= 0.0
                    && self.takeoff_drive_elapsed_seconds
                        >= self.parameters.takeoff_drive_dwell_seconds =>
            {
                self.enter(FlightMode::Takeoff)
            }
            FlightMode::Takeoff
                if input.root_height_mm >= 9.0 || self.mode_elapsed_seconds >= 0.8 =>
            {
                self.enter(FlightMode::Cruise)
            }
            FlightMode::Takeoff | FlightMode::Cruise if input.landing_request => {
                self.enter(FlightMode::Landing)
            }
            FlightMode::Cruise
                if self.surface_contact_elapsed_seconds >= SURFACE_CONTACT_DWELL_SECONDS
                    && !input.brain_enabled
                    && !input.collision_escape_active
                    && input.brain_flight_drive < self.parameters.takeoff_drive_threshold =>
            {
                self.enter(FlightMode::Grounded)
            }
            FlightMode::Cruise
                if !input.brain_enabled
                    && odor_total >= self.parameters.landing_odor_threshold
                    && (input.odor_left - input.odor_right).abs()
                        <= self.parameters.landing_bilateral_fraction * odor_total =>
            {
                self.enter(FlightMode::Landing)
            }
            FlightMode::Landing
                if input.contact_count >= 3
                    && self.landing_load_transfer >= 1.0
                    && input.angular_speed_rad_s < 5.0
                    && input.vertical_velocity_mm_s.abs() < 10.0 =>
            {
                self.enter(FlightMode::Grounded)
            }
            _ => {}
        }

        let odor_steering = if odor_total > 1e-9 {
            (self.parameters.odor_gradient_gain * (input.odor_left - input.odor_right) / odor_total)
                .clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let odor_steering_contribution = if input.brain_enabled {
            0.0
        } else {
            self.parameters.odor_steering_weight * odor_steering
        };
        let wander_steering_contribution = if input.brain_enabled {
            0.0
        } else {
            self.parameters.wander_steering_weight * self.wander
        };
        let brain_steering_contribution = if input.brain_enabled {
            input.brain_steering
        } else {
            self.parameters.brain_steering_weight * input.brain_steering
        };
        let steering = if input.brain_enabled {
            brain_steering_contribution.clamp(
                -self.parameters.maximum_abs_steering,
                self.parameters.maximum_abs_steering,
            )
        } else {
            (odor_steering_contribution
                + wander_steering_contribution
                + brain_steering_contribution)
                .clamp(
                    -self.parameters.maximum_abs_steering,
                    self.parameters.maximum_abs_steering,
                )
        };
        let mut altitude_target_clamped = false;
        let mut neural_altitude_contribution_mm_s = 0.0;
        let mut optic_flow_altitude_contribution_mm_s = 0.0;
        let neural_altitude_motor_rate_mm_s = if input.cns_motor_activation.is_some() {
            self.parameters.neural_altitude_rate_mm_s * input.brain_altitude_control
        } else if matches!(self.mode, FlightMode::Takeoff | FlightMode::Cruise) {
            self.neural_altitude_command(input)
        } else {
            0.0
        };
        if matches!(self.mode, FlightMode::Takeoff | FlightMode::Cruise) && !input.altitude_hold {
            self.neural_altitude_bout_remaining_seconds =
                (self.neural_altitude_bout_remaining_seconds - input.dt_seconds).max(0.0);
            if self.neural_altitude_bout_remaining_seconds == 0.0 {
                self.neural_altitude_motor_command = 0.0;
            }
        }
        if matches!(self.mode, FlightMode::Takeoff | FlightMode::Cruise) {
            if let Some(power) = input.cns_motor_activation {
                let bounds = input.flight_altitude_bounds_mm;
                let power_target = if input.cns_food_approach {
                    input
                        .cns_approach_height_mm
                        .unwrap_or(self.parameters.cruise_height_mm)
                } else {
                    self.parameters.takeoff_height_mm
                        + power * (bounds[1] - self.parameters.takeoff_height_mm).max(0.0)
                };
                if !input.altitude_hold {
                    let maximum_change =
                        self.parameters.neural_altitude_rate_mm_s * input.dt_seconds;
                    self.cruise_target_height_mm += (power_target - self.cruise_target_height_mm)
                        .clamp(-maximum_change, maximum_change);
                }
            } else if input.brain_enabled {
                self.airborne_peak_flight_drive = self
                    .airborne_peak_flight_drive
                    .max(input.brain_flight_drive);
                if self.neural_altitude_motor_command == 0.0 && !input.altitude_hold {
                    self.cruise_target_height_mm = self
                        .cruise_target_height_mm
                        .max(self.flight_drive_altitude_target(input.flight_altitude_bounds_mm));
                }
            }
            neural_altitude_contribution_mm_s = if input.cns_food_approach {
                0.0
            } else {
                neural_altitude_motor_rate_mm_s
            };
            if self.mode == FlightMode::Cruise {
                optic_flow_altitude_contribution_mm_s = if input.brain_enabled {
                    0.0
                } else {
                    self.parameters.optic_flow_altitude_rate_mm_s
                        * control_deadband(input.optic_flow_altitude_control, 0.08)
                };
            }
            if input.altitude_hold {
                neural_altitude_contribution_mm_s = 0.0;
                optic_flow_altitude_contribution_mm_s = 0.0;
            }
            let requested_target = self.cruise_target_height_mm
                + (neural_altitude_contribution_mm_s + optic_flow_altitude_contribution_mm_s)
                    * input.dt_seconds;
            self.cruise_target_height_mm = requested_target.clamp(
                input.flight_altitude_bounds_mm[0],
                input.flight_altitude_bounds_mm[1],
            );
            altitude_target_clamped =
                (requested_target - self.cruise_target_height_mm).abs() > f64::EPSILON;
        }
        let (target_height_mm, base_amplitude) = match self.mode {
            FlightMode::Grounded => (input.root_height_mm, 0.0),
            FlightMode::Takeoff => (
                if input.cns_motor_activation.is_some() {
                    self.cruise_target_height_mm
                } else {
                    self.parameters.takeoff_height_mm
                },
                self.parameters.takeoff_amplitude,
            ),
            FlightMode::Cruise => (
                self.cruise_target_height_mm,
                self.parameters.cruise_amplitude,
            ),
            FlightMode::Landing => (
                self.parameters.landing_height_mm,
                self.parameters.landing_amplitude,
            ),
        };
        let amplitude_scale = if self.mode == FlightMode::Grounded {
            0.0
        } else if self.mode == FlightMode::Landing {
            (base_amplitude + self.parameters.brain_amplitude_gain * input.brain_flight_drive)
                .clamp(0.35, 1.0)
                * (1.0 - self.landing_load_transfer)
        } else {
            (base_amplitude
                + self.parameters.brain_amplitude_gain * input.brain_flight_drive
                + 0.012 * (target_height_mm - input.root_height_mm)
                - 0.003 * input.vertical_velocity_mm_s)
                .clamp(0.35, 1.0)
        };
        Ok(FlightBehaviorCommand {
            mode: self.mode,
            amplitude_scale,
            steering,
            target_height_mm,
            altitude_target_clamped,
            odor_steering_contribution,
            wander_steering_contribution,
            brain_steering_contribution,
            neural_altitude_contribution_mm_s,
            optic_flow_altitude_contribution_mm_s,
        })
    }

    fn enter(&mut self, mode: FlightMode) {
        if self.mode != mode {
            self.mode = mode;
            self.mode_elapsed_seconds = 0.0;
            self.landing_load_transfer = 0.0;
            self.landing_contact_lost_seconds = 0.0;
            if mode != FlightMode::Grounded {
                self.takeoff_drive_elapsed_seconds = 0.0;
            }
            if mode == FlightMode::Grounded {
                self.surface_contact_elapsed_seconds = 0.0;
                self.airborne_peak_flight_drive = 0.0;
            }
            if matches!(mode, FlightMode::Grounded | FlightMode::Landing) {
                self.neural_altitude_input_active = false;
                self.neural_altitude_motor_command = 0.0;
                self.neural_altitude_bout_remaining_seconds = 0.0;
            }
        }
    }

    fn neural_altitude_command(&mut self, input: FlightBehaviorInput) -> f64 {
        let raw = input.brain_altitude_control;
        let changed_direction = self.neural_altitude_motor_command != 0.0
            && raw.abs() >= NEURAL_ALTITUDE_ACTIVATION
            && raw.signum() != self.neural_altitude_motor_command.signum();
        let bout_expired = self.neural_altitude_bout_remaining_seconds <= 0.0;
        if raw.abs() >= NEURAL_ALTITUDE_ACTIVATION
            && (!self.neural_altitude_input_active || changed_direction || bout_expired)
        {
            let magnitude = (NEURAL_ALTITUDE_MOTOR_GAIN
                * control_deadband(raw.abs(), NEURAL_ALTITUDE_RELEASE))
            .clamp(NEURAL_ALTITUDE_MINIMUM_COMMAND, 1.0);
            self.neural_altitude_motor_command = raw.signum() * magnitude;
            self.neural_altitude_bout_remaining_seconds = NEURAL_ALTITUDE_BOUT_SECONDS;
            self.neural_altitude_input_active = true;
        } else if raw.abs() <= NEURAL_ALTITUDE_RELEASE {
            self.neural_altitude_input_active = false;
        }
        self.parameters.neural_altitude_rate_mm_s * self.neural_altitude_motor_command
    }

    fn flight_drive_altitude_target(&self, bounds_mm: [f64; 2]) -> f64 {
        let neutral_height_mm = self
            .parameters
            .cruise_height_mm
            .clamp(bounds_mm[0], bounds_mm[1]);
        let maximum_height_mm = neutral_height_mm
            + NEURAL_FLIGHT_DRIVE_MAXIMUM_HEIGHT_FRACTION * (bounds_mm[1] - neutral_height_mm);
        let drive = ((self.airborne_peak_flight_drive - self.parameters.takeoff_drive_threshold)
            / NEURAL_FLIGHT_DRIVE_ALTITUDE_RANGE)
            .clamp(0.0, 1.0);
        neutral_height_mm + drive * (maximum_height_mm - neutral_height_mm)
    }

    fn next_unit(&mut self) -> f64 {
        self.random_state = self
            .random_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        f64::from((self.random_state >> 32) as u32) / f64::from(u32::MAX)
    }
}

fn validate_input(input: FlightBehaviorInput) -> Result<()> {
    if !input.dt_seconds.is_finite()
        || input.dt_seconds <= 0.0
        || !input.root_height_mm.is_finite()
        || !input.vertical_velocity_mm_s.is_finite()
        || !input.angular_speed_rad_s.is_finite()
        || input.angular_speed_rad_s < 0.0
        || !input.odor_left.is_finite()
        || input.odor_left < 0.0
        || !input.odor_right.is_finite()
        || input.odor_right < 0.0
        || !input.taste_valence.is_finite()
        || !(-1.0..=1.0).contains(&input.taste_valence)
        || !input.brain_flight_drive.is_finite()
        || input
            .cns_motor_activation
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        || !input.brain_steering.is_finite()
        || input
            .cns_approach_height_mm
            .is_some_and(|height| !height.is_finite() || height <= 0.0)
        || !input.brain_altitude_control.is_finite()
        || !input.optic_flow_altitude_control.is_finite()
        || !(-1.0..=1.0).contains(&input.brain_flight_drive)
        || !(-1.0..=1.0).contains(&input.brain_steering)
        || !(-1.0..=1.0).contains(&input.brain_altitude_control)
        || !(-1.0..=1.0).contains(&input.optic_flow_altitude_control)
        || input
            .flight_altitude_bounds_mm
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || input.flight_altitude_bounds_mm[0] >= input.flight_altitude_bounds_mm[1]
    {
        bail!("flight behavior input is invalid")
    }
    Ok(())
}

fn control_deadband(value: f64, deadband: f64) -> f64 {
    if value.abs() <= deadband {
        0.0
    } else {
        value.signum() * (value.abs() - deadband) / (1.0 - deadband)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cns_output_disconnect_prevents_takeoff_despite_descending_drive() {
        let mut controller = FlightBehaviorController::new(1);
        for _ in 0..100 {
            let command = controller
                .update(FlightBehaviorInput {
                    dt_seconds: 0.01,
                    enabled: true,
                    brain_enabled: true,
                    brain_flight_drive: 1.0,
                    cns_motor_activation: Some(0.0),
                    ..FlightBehaviorInput::default()
                })
                .unwrap();
            assert_eq!(command.mode, FlightMode::Grounded);
            assert_eq!(command.amplitude_scale, 0.0);
        }
    }

    #[test]
    fn cns_altitude_tracks_current_motor_power_not_historical_peak() {
        let mut controller = FlightBehaviorController::new(1);
        controller.enter(FlightMode::Cruise);
        let mut target = 0.0;
        for _ in 0..600 {
            target = controller
                .update(FlightBehaviorInput {
                    dt_seconds: 0.01,
                    enabled: true,
                    brain_enabled: true,
                    root_height_mm: 50.0,
                    brain_flight_drive: 0.8,
                    cns_motor_activation: Some(0.8),
                    ..FlightBehaviorInput::default()
                })
                .unwrap()
                .target_height_mm;
        }
        let high_target = target;
        for _ in 0..600 {
            target = controller
                .update(FlightBehaviorInput {
                    dt_seconds: 0.01,
                    enabled: true,
                    brain_enabled: true,
                    root_height_mm: 50.0,
                    brain_flight_drive: 0.2,
                    cns_motor_activation: Some(0.2),
                    ..FlightBehaviorInput::default()
                })
                .unwrap()
                .target_height_mm;
        }
        assert!(high_target > 150.0);
        assert!(target < 70.0);
    }

    #[test]
    fn cns_food_approach_arbitrates_altitude_without_silencing_wings() {
        let mut controller = FlightBehaviorController::new(1);
        controller.enter(FlightMode::Cruise);
        controller.cruise_target_height_mm = 185.0;
        let input = FlightBehaviorInput {
            dt_seconds: 0.01,
            enabled: true,
            brain_enabled: true,
            root_height_mm: 28.0,
            brain_flight_drive: 0.9,
            cns_motor_activation: Some(0.9),
            brain_altitude_control: 0.8,
            cns_food_approach: true,
            ..FlightBehaviorInput::default()
        };
        let mut command = FlightBehaviorCommand::default();
        for _ in 0..500 {
            command = controller.update(input).unwrap();
        }
        assert_eq!(command.mode, FlightMode::Cruise);
        assert!((command.target_height_mm - 28.0).abs() < 1e-9);
        assert!(command.amplitude_scale > 0.9);
        assert_eq!(command.neural_altitude_contribution_mm_s, 0.0);
        let released = controller
            .update(FlightBehaviorInput {
                cns_food_approach: false,
                ..input
            })
            .unwrap();
        assert!(released.target_height_mm > command.target_height_mm);
    }

    #[test]
    fn landing_contact_loss_restores_wing_support() {
        let mut controller = FlightBehaviorController::new(1);
        controller.enter(FlightMode::Landing);
        let input = FlightBehaviorInput {
            dt_seconds: 0.04,
            enabled: true,
            brain_enabled: true,
            root_height_mm: 5.0,
            ..Default::default()
        };
        let airborne = controller.update(input).unwrap();
        let single_foot = controller
            .update(FlightBehaviorInput {
                contact_count: 1,
                ..input
            })
            .unwrap();
        assert_eq!(single_foot.amplitude_scale, airborne.amplitude_scale);
        let grazing = controller
            .update(FlightBehaviorInput {
                contact_count: 2,
                ..input
            })
            .unwrap();
        assert_eq!(grazing.mode, FlightMode::Landing);
        assert!(grazing.amplitude_scale < airborne.amplitude_scale);
        let released = controller.update(input).unwrap();
        assert_eq!(released.mode, FlightMode::Landing);
        assert_eq!(released.amplitude_scale, airborne.amplitude_scale);
    }

    #[test]
    fn touchdown_contact_chatter_does_not_keep_wings_flapping() {
        let mut controller = FlightBehaviorController::new(1);
        controller.enter(FlightMode::Landing);
        let input = FlightBehaviorInput {
            dt_seconds: 0.002,
            enabled: true,
            brain_enabled: true,
            brain_flight_drive: 0.9,
            cns_motor_activation: Some(0.9),
            angular_speed_rad_s: 10.0,
            ..Default::default()
        };
        let mut command = FlightBehaviorCommand::default();
        for window in 0..60 {
            command = controller
                .update(FlightBehaviorInput {
                    contact_count: if window % 3 == 0 { 2 } else { 0 },
                    ..input
                })
                .unwrap();
        }
        assert_eq!(command.mode, FlightMode::Landing);
        assert_eq!(command.amplitude_scale, 0.0);
        for _ in 0..60 {
            command = controller.update(input).unwrap();
        }
        assert_eq!(command.mode, FlightMode::Landing);
        assert!(command.amplitude_scale > 0.9);
    }

    #[test]
    fn landing_request_interrupts_cns_takeoff_and_requires_physical_touchdown() {
        let mut controller = FlightBehaviorController::new(1);
        controller.enter(FlightMode::Takeoff);
        let input = FlightBehaviorInput {
            dt_seconds: 0.01,
            enabled: true,
            brain_enabled: true,
            root_height_mm: 5.0,
            brain_flight_drive: 0.9,
            cns_motor_activation: Some(0.9),
            landing_request: true,
            ..FlightBehaviorInput::default()
        };
        for _ in 0..100 {
            let command = controller.update(input).unwrap();
            assert_eq!(command.mode, FlightMode::Landing);
            assert_eq!(command.target_height_mm, 0.6);
        }
        let single_foot = controller
            .update(FlightBehaviorInput {
                contact_count: 1,
                ..input
            })
            .unwrap();
        assert_eq!(single_foot.mode, FlightMode::Landing);
        for _ in 0..10 {
            let partial = controller
                .update(FlightBehaviorInput {
                    contact_count: 2,
                    ..input
                })
                .unwrap();
            assert_eq!(partial.mode, FlightMode::Landing);
        }
        let moving = controller
            .update(FlightBehaviorInput {
                contact_count: 6,
                angular_speed_rad_s: 10.0,
                ..input
            })
            .unwrap();
        assert_eq!(moving.mode, FlightMode::Landing);
        controller
            .update(FlightBehaviorInput {
                contact_count: 6,
                ..input
            })
            .unwrap();
        assert_eq!(controller.mode, FlightMode::Grounded);
    }

    fn input() -> FlightBehaviorInput {
        FlightBehaviorInput {
            dt_seconds: 0.05,
            enabled: true,
            root_height_mm: 1.0,
            contact_count: 6,
            ..FlightBehaviorInput::default()
        }
    }

    fn enter_cruise(controller: &mut FlightBehaviorController) -> FlightBehaviorCommand {
        let driven = FlightBehaviorInput {
            brain_flight_drive: TAKEOFF_DRIVE_THRESHOLD + 0.01,
            ..input()
        };
        for _ in 0..3 {
            controller.update(driven).unwrap();
        }
        controller
            .update(FlightBehaviorInput {
                root_height_mm: 10.0,
                ..driven
            })
            .unwrap()
    }

    #[test]
    fn grounded_fly_takes_off_only_after_sustained_brain_drive() {
        let mut controller = FlightBehaviorController::new(5);
        for _ in 0..20 {
            assert_eq!(
                controller.update(input()).unwrap().mode,
                FlightMode::Grounded
            );
        }
        let driven = FlightBehaviorInput {
            brain_flight_drive: TAKEOFF_DRIVE_THRESHOLD + 0.01,
            ..input()
        };
        assert_eq!(
            controller.update(driven).unwrap().mode,
            FlightMode::Grounded
        );
        assert_eq!(
            controller.update(driven).unwrap().mode,
            FlightMode::Grounded
        );
        let command = controller.update(driven).unwrap();
        assert_eq!(command.mode, FlightMode::Takeoff);
        assert_eq!(command.amplitude_scale, 1.0);
    }

    #[test]
    fn interrupted_brain_drive_does_not_accumulate_toward_takeoff() {
        let mut controller = FlightBehaviorController::new(6);
        let driven = FlightBehaviorInput {
            brain_flight_drive: TAKEOFF_DRIVE_THRESHOLD + 0.01,
            ..input()
        };
        for _ in 0..10 {
            assert_eq!(
                controller.update(driven).unwrap().mode,
                FlightMode::Grounded
            );
            assert_eq!(
                controller.update(input()).unwrap().mode,
                FlightMode::Grounded
            );
        }
    }

    #[test]
    fn balanced_strong_odor_triggers_landing_after_takeoff() {
        let mut controller = FlightBehaviorController::new(7);
        for _ in 0..3 {
            controller
                .update(FlightBehaviorInput {
                    brain_flight_drive: TAKEOFF_DRIVE_THRESHOLD + 0.01,
                    ..input()
                })
                .unwrap();
        }
        let mut airborne = input();
        airborne.root_height_mm = 10.0;
        airborne.contact_count = 0;
        assert_eq!(
            controller.update(airborne).unwrap().mode,
            FlightMode::Cruise
        );
        airborne.odor_left = 0.7;
        airborne.odor_right = 0.65;
        assert_eq!(
            controller.update(airborne).unwrap().mode,
            FlightMode::Landing
        );
    }

    #[test]
    fn full_brain_mode_does_not_bypass_the_connectome_for_odor_landing() {
        let mut controller = FlightBehaviorController::new(17);
        enter_cruise(&mut controller);
        let command = controller
            .update(FlightBehaviorInput {
                enabled: true,
                brain_enabled: true,
                dt_seconds: 0.05,
                root_height_mm: 28.0,
                odor_left: 0.7,
                odor_right: 0.65,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(command.mode, FlightMode::Cruise);
    }

    #[test]
    fn full_brain_incidental_surface_contact_does_not_ground_the_fly() {
        let mut controller = FlightBehaviorController::new(21);
        enter_cruise(&mut controller);
        let contact = controller
            .update(FlightBehaviorInput {
                enabled: true,
                brain_enabled: true,
                dt_seconds: 0.05,
                root_height_mm: 80.0,
                contact_count: 2,
                brain_flight_drive: 0.0,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(contact.mode, FlightMode::Cruise);

        let requested = controller
            .update(FlightBehaviorInput {
                enabled: true,
                brain_enabled: true,
                landing_request: true,
                dt_seconds: 0.05,
                root_height_mm: 80.0,
                contact_count: 2,
                brain_flight_drive: 0.0,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(requested.mode, FlightMode::Landing);

        let grounded = controller
            .update(FlightBehaviorInput {
                enabled: true,
                brain_enabled: true,
                dt_seconds: 0.1,
                root_height_mm: 80.0,
                contact_count: 6,
                brain_flight_drive: 0.0,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(grounded.mode, FlightMode::Grounded);
    }

    #[test]
    fn landing_request_transitions_full_brain_flight_to_landing() {
        let mut controller = FlightBehaviorController::new(18);
        enter_cruise(&mut controller);
        let command = controller
            .update(FlightBehaviorInput {
                enabled: true,
                brain_enabled: true,
                landing_request: true,
                dt_seconds: 0.05,
                root_height_mm: 28.0,
                brain_flight_drive: 0.5,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(command.mode, FlightMode::Landing);
        assert_eq!(command.target_height_mm, 0.6);
    }

    #[test]
    fn flight_resolves_a_small_antennal_plume_difference() {
        let mut controller = FlightBehaviorController::new(12);
        enter_cruise(&mut controller);
        let command = controller
            .update(FlightBehaviorInput {
                enabled: true,
                dt_seconds: 0.002,
                root_height_mm: 30.0,
                odor_left: 0.0174,
                odor_right: 0.0172,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert!(command.odor_steering_contribution > 0.1);
    }

    #[test]
    fn full_brain_steering_uses_decoded_flight_dn_without_procedural_terms() {
        let mut controller = FlightBehaviorController::new(16);
        enter_cruise(&mut controller);
        let command = controller
            .update(FlightBehaviorInput {
                enabled: true,
                brain_enabled: true,
                dt_seconds: 0.05,
                root_height_mm: 28.0,
                odor_left: 1.0,
                odor_right: 0.0,
                brain_steering: 0.4,
                optic_flow_altitude_control: 1.0,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(command.odor_steering_contribution, 0.0);
        assert_eq!(command.wander_steering_contribution, 0.0);
        assert_eq!(command.brain_steering_contribution, 0.4);
        assert_eq!(command.steering, command.brain_steering_contribution);
        assert_eq!(command.optic_flow_altitude_contribution_mm_s, 0.0);
        assert_eq!(command.target_height_mm, 28.0);
    }

    #[test]
    fn low_flight_drive_allows_touchdown_on_an_elevated_surface() {
        let mut controller = FlightBehaviorController::new(8);
        enter_cruise(&mut controller);
        let command = controller
            .update(FlightBehaviorInput {
                enabled: true,
                dt_seconds: 0.05,
                root_height_mm: 80.0,
                contact_count: 2,
                brain_flight_drive: 0.0,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(command.mode, FlightMode::Grounded);
        assert_eq!(command.amplitude_scale, 0.0);
    }

    #[test]
    fn takeoff_inhibited_blocks_sustained_neural_drive() {
        let mut controller = FlightBehaviorController::new(19);
        let inhibited = FlightBehaviorInput {
            takeoff_inhibited: true,
            brain_flight_drive: TAKEOFF_DRIVE_THRESHOLD + 0.2,
            ..input()
        };
        for _ in 0..20 {
            let command = controller.update(inhibited).unwrap();
            assert_eq!(command.mode, FlightMode::Grounded);
            assert_eq!(command.amplitude_scale, 0.0);
        }

        let released = FlightBehaviorInput {
            takeoff_inhibited: false,
            brain_flight_drive: TAKEOFF_DRIVE_THRESHOLD + 0.2,
            ..input()
        };
        assert_eq!(
            controller.update(released).unwrap().mode,
            FlightMode::Grounded
        );
        assert_eq!(
            controller.update(released).unwrap().mode,
            FlightMode::Grounded
        );
        assert_eq!(
            controller.update(released).unwrap().mode,
            FlightMode::Takeoff
        );
    }

    #[test]
    fn strong_flight_drive_rejects_an_incidental_surface_contact() {
        let mut controller = FlightBehaviorController::new(9);
        enter_cruise(&mut controller);
        let command = controller
            .update(FlightBehaviorInput {
                enabled: true,
                dt_seconds: 0.05,
                root_height_mm: 80.0,
                contact_count: 2,
                brain_flight_drive: TAKEOFF_DRIVE_THRESHOLD + 0.01,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(command.mode, FlightMode::Cruise);
    }

    #[test]
    fn brief_low_drive_surface_contact_does_not_count_as_a_landing() {
        let mut controller = FlightBehaviorController::new(10);
        enter_cruise(&mut controller);
        let contact = controller
            .update(FlightBehaviorInput {
                enabled: true,
                dt_seconds: 0.01,
                root_height_mm: 80.0,
                contact_count: 2,
                brain_flight_drive: 0.0,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(contact.mode, FlightMode::Cruise);
        let released = controller
            .update(FlightBehaviorInput {
                enabled: true,
                dt_seconds: 0.05,
                root_height_mm: 80.0,
                contact_count: 0,
                brain_flight_drive: 0.0,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(released.mode, FlightMode::Cruise);
    }

    #[test]
    fn disabling_flight_stops_wing_amplitude_immediately() {
        let mut controller = FlightBehaviorController::new(11);
        for _ in 0..3 {
            controller
                .update(FlightBehaviorInput {
                    brain_flight_drive: TAKEOFF_DRIVE_THRESHOLD + 0.01,
                    ..input()
                })
                .unwrap();
        }
        let command = controller
            .update(FlightBehaviorInput {
                enabled: false,
                ..input()
            })
            .unwrap();
        assert_eq!(command.mode, FlightMode::Grounded);
        assert_eq!(command.amplitude_scale, 0.0);
    }

    #[test]
    fn neural_altitude_control_integrates_a_bounded_cruise_target() {
        let mut level = FlightBehaviorController::new(12);
        let level_command = enter_cruise(&mut level);
        assert_eq!(level_command.mode, FlightMode::Cruise);
        assert_eq!(level_command.target_height_mm, 28.0);

        let mut climbing = FlightBehaviorController::new(12);
        enter_cruise(&mut climbing);
        let mut climb_command = FlightBehaviorCommand::default();
        for _ in 0..10 {
            climb_command = climbing
                .update(FlightBehaviorInput {
                    dt_seconds: 0.05,
                    enabled: true,
                    root_height_mm: 28.0,
                    brain_altitude_control: 1.0,
                    ..FlightBehaviorInput::default()
                })
                .unwrap();
        }
        assert_eq!(climb_command.target_height_mm, 48.0);

        let mut descending = FlightBehaviorController::new(12);
        enter_cruise(&mut descending);
        let mut descend_command = FlightBehaviorCommand::default();
        for _ in 0..10 {
            descend_command = descending
                .update(FlightBehaviorInput {
                    dt_seconds: 0.05,
                    enabled: true,
                    root_height_mm: 28.0,
                    brain_altitude_control: -1.0,
                    ..FlightBehaviorInput::default()
                })
                .unwrap();
        }
        assert_eq!(descend_command.target_height_mm, 8.0);

        let clamped = climbing
            .update(FlightBehaviorInput {
                dt_seconds: 1.0,
                enabled: true,
                root_height_mm: 38.0,
                brain_altitude_control: 1.0,
                flight_altitude_bounds_mm: [5.0, 40.0],
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(clamped.target_height_mm, 40.0);
        assert!(clamped.altitude_target_clamped);

        climbing.reset(12);
        assert_eq!(enter_cruise(&mut climbing).target_height_mm, 28.0);
    }

    #[test]
    fn isolated_neural_altitude_pulse_persists_and_stays_bounded() {
        let mut controller = FlightBehaviorController::new(20);
        let initial = enter_cruise(&mut controller);
        assert_eq!(initial.target_height_mm, 28.0);

        let pulse = controller
            .update(FlightBehaviorInput {
                dt_seconds: 0.05,
                enabled: true,
                root_height_mm: 28.0,
                brain_altitude_control: 1.0,
                flight_altitude_bounds_mm: [5.0, 40.0],
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert!(pulse.target_height_mm > initial.target_height_mm + 1.0);
        assert!(pulse.target_height_mm <= 40.0);

        let mut target = pulse.target_height_mm;
        let mut clamped = false;
        for _ in 0..20 {
            let command = controller
                .update(FlightBehaviorInput {
                    dt_seconds: 0.05,
                    enabled: true,
                    root_height_mm: 28.0,
                    brain_altitude_control: 0.0,
                    flight_altitude_bounds_mm: [5.0, 40.0],
                    ..FlightBehaviorInput::default()
                })
                .unwrap();
            assert!(command.target_height_mm >= target);
            assert!(command.target_height_mm <= 40.0);
            target = command.target_height_mm;
            clamped |= command.altitude_target_clamped;
        }
        assert!(target - initial.target_height_mm >= 10.0);
        assert_eq!(target, 40.0);
        assert!(clamped);
    }

    #[test]
    fn weak_neural_altitude_pulse_changes_the_bounded_cruise_target() {
        let mut controller = FlightBehaviorController::new(22);
        let initial = enter_cruise(&mut controller);
        let command = controller
            .update(FlightBehaviorInput {
                dt_seconds: 0.05,
                enabled: true,
                brain_enabled: true,
                root_height_mm: 28.0,
                brain_altitude_control: 0.03,
                flight_altitude_bounds_mm: [5.0, 40.0],
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert!(command.neural_altitude_contribution_mm_s > 0.0);
        assert!(command.target_height_mm > initial.target_height_mm);
    }

    #[test]
    fn sustained_neural_altitude_drive_renews_motor_intent_bouts() {
        let mut controller = FlightBehaviorController::new(23);
        let initial = enter_cruise(&mut controller);
        let mut command = initial;
        for _ in 0..50 {
            command = controller
                .update(FlightBehaviorInput {
                    dt_seconds: 0.05,
                    enabled: true,
                    brain_enabled: true,
                    root_height_mm: 28.0,
                    brain_altitude_control: 1.0,
                    flight_altitude_bounds_mm: [5.0, 208.0],
                    ..FlightBehaviorInput::default()
                })
                .unwrap();
        }
        assert!(command.target_height_mm > initial.target_height_mm + 90.0);
        assert!(command.target_height_mm <= 208.0);
    }

    #[test]
    fn full_brain_flight_drive_selects_a_higher_bounded_altitude() {
        let mut controller = FlightBehaviorController::new(26);
        let driven = FlightBehaviorInput {
            dt_seconds: 0.05,
            enabled: true,
            brain_enabled: true,
            root_height_mm: 1.0,
            contact_count: 6,
            brain_flight_drive: 0.06,
            flight_altitude_bounds_mm: [5.0, 208.0],
            ..FlightBehaviorInput::default()
        };
        for _ in 0..3 {
            controller.update(driven).unwrap();
        }
        let cruise = controller
            .update(FlightBehaviorInput {
                root_height_mm: 10.0,
                contact_count: 0,
                ..driven
            })
            .unwrap();
        assert_eq!(cruise.mode, FlightMode::Cruise);
        assert!(cruise.target_height_mm > 100.0);
        assert!(cruise.target_height_mm < 208.0);

        let retained = controller
            .update(FlightBehaviorInput {
                brain_flight_drive: 0.0,
                ..driven
            })
            .unwrap();
        assert_eq!(retained.target_height_mm, cruise.target_height_mm);
    }

    #[test]
    fn neural_altitude_intent_started_during_takeoff_reaches_cruise_target() {
        let mut controller = FlightBehaviorController::new(24);
        let driven = FlightBehaviorInput {
            brain_enabled: true,
            brain_flight_drive: TAKEOFF_DRIVE_THRESHOLD + 0.01,
            brain_altitude_control: 0.03,
            ..input()
        };
        for _ in 0..3 {
            controller.update(driven).unwrap();
        }
        let command = controller
            .update(FlightBehaviorInput {
                root_height_mm: 10.0,
                brain_altitude_control: 0.0,
                ..driven
            })
            .unwrap();
        assert_eq!(command.mode, FlightMode::Cruise);
        assert!(command.target_height_mm > controller.parameters.cruise_height_mm + 1.0);
    }

    #[test]
    fn neural_cruise_target_survives_landing_and_takeoff() {
        let mut controller = FlightBehaviorController::new(25);
        enter_cruise(&mut controller);
        let raised = controller
            .update(FlightBehaviorInput {
                enabled: true,
                brain_enabled: true,
                dt_seconds: 0.05,
                root_height_mm: 28.0,
                brain_altitude_control: 1.0,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert!(raised.target_height_mm > 28.0);

        let landing = controller
            .update(FlightBehaviorInput {
                enabled: true,
                brain_enabled: true,
                landing_request: true,
                dt_seconds: 0.05,
                root_height_mm: 28.0,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(landing.mode, FlightMode::Landing);
        assert_eq!(
            controller
                .update(FlightBehaviorInput {
                    enabled: true,
                    brain_enabled: true,
                    dt_seconds: 0.1,
                    root_height_mm: 28.0,
                    contact_count: 6,
                    ..FlightBehaviorInput::default()
                })
                .unwrap()
                .mode,
            FlightMode::Grounded
        );

        let driven = FlightBehaviorInput {
            enabled: true,
            brain_enabled: true,
            brain_flight_drive: TAKEOFF_DRIVE_THRESHOLD + 0.01,
            ..input()
        };
        for _ in 0..3 {
            controller.update(driven).unwrap();
        }
        let takeoff = controller
            .update(FlightBehaviorInput {
                root_height_mm: 10.0,
                ..driven
            })
            .unwrap();
        assert_eq!(takeoff.mode, FlightMode::Cruise);
        assert!(takeoff.target_height_mm > 28.0);
    }

    #[test]
    fn landing_ignores_neural_altitude_control() {
        let mut controller = FlightBehaviorController::new(13);
        enter_cruise(&mut controller);
        let command = controller
            .update(FlightBehaviorInput {
                dt_seconds: 0.05,
                enabled: true,
                root_height_mm: 28.0,
                odor_left: 0.7,
                odor_right: 0.65,
                brain_altitude_control: 1.0,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(command.mode, FlightMode::Landing);
        assert_eq!(command.target_height_mm, 0.6);
    }

    #[test]
    fn optic_flow_and_neural_altitude_contributions_are_separate() {
        let mut controller = FlightBehaviorController::new(14);
        enter_cruise(&mut controller);
        let command = controller
            .update(FlightBehaviorInput {
                dt_seconds: 0.5,
                enabled: true,
                root_height_mm: 28.0,
                brain_altitude_control: 0.5,
                optic_flow_altitude_control: 1.0,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert!(command.neural_altitude_contribution_mm_s > 0.0);
        assert_eq!(command.optic_flow_altitude_contribution_mm_s, 24.0);
        assert!(command.target_height_mm > 40.0);
    }

    #[test]
    fn overhead_hold_freezes_both_altitude_contributions() {
        let mut controller = FlightBehaviorController::new(15);
        enter_cruise(&mut controller);
        let command = controller
            .update(FlightBehaviorInput {
                dt_seconds: 0.5,
                enabled: true,
                root_height_mm: 28.0,
                brain_altitude_control: 1.0,
                optic_flow_altitude_control: 1.0,
                altitude_hold: true,
                ..FlightBehaviorInput::default()
            })
            .unwrap();
        assert_eq!(command.target_height_mm, 28.0);
        assert_eq!(command.neural_altitude_contribution_mm_s, 0.0);
        assert_eq!(command.optic_flow_altitude_contribution_mm_s, 0.0);
    }

    #[test]
    fn bilateral_odor_taxis_improves_airborne_target_approach() {
        fn run(with_odor: bool) -> f64 {
            const DT_SECONDS: f64 = 0.01;
            let target = [24.0, 12.0];
            let mut controller = FlightBehaviorController::new(0x1234);
            let drive = TAKEOFF_DRIVE_THRESHOLD + 0.01;
            for _ in 0..15 {
                controller
                    .update(FlightBehaviorInput {
                        dt_seconds: DT_SECONDS,
                        enabled: true,
                        root_height_mm: 20.0,
                        brain_flight_drive: drive,
                        ..FlightBehaviorInput::default()
                    })
                    .unwrap();
            }
            let mut position = [0.0, 0.0];
            let mut heading = [1.0, 0.0];
            for _ in 0..600 {
                let left_sensor = [
                    position[0] - 0.5 * heading[1],
                    position[1] + 0.5 * heading[0],
                ];
                let right_sensor = [
                    position[0] + 0.5 * heading[1],
                    position[1] - 0.5 * heading[0],
                ];
                let odor = |sensor: [f64; 2]| {
                    let distance = (target[0] - sensor[0]).hypot(target[1] - sensor[1]);
                    if with_odor {
                        (-distance / 25.0).exp()
                    } else {
                        0.0
                    }
                };
                let command = controller
                    .update(FlightBehaviorInput {
                        dt_seconds: DT_SECONDS,
                        enabled: true,
                        root_height_mm: 20.0,
                        odor_left: odor(left_sensor),
                        odor_right: odor(right_sensor),
                        ..FlightBehaviorInput::default()
                    })
                    .unwrap();
                assert_ne!(command.mode, FlightMode::Grounded);
                let turn = command.steering * 2.0 * DT_SECONDS;
                heading = [
                    turn.cos() * heading[0] - turn.sin() * heading[1],
                    turn.sin() * heading[0] + turn.cos() * heading[1],
                ];
                position[0] += 4.0 * DT_SECONDS * heading[0];
                position[1] += 4.0 * DT_SECONDS * heading[1];
            }
            (target[0] - position[0]).hypot(target[1] - position[1])
        }

        let without_odor = run(false);
        let with_odor = run(true);
        assert!(
            with_odor + 0.5 < without_odor,
            "odor taxis did not improve target approach: with={with_odor}, without={without_odor}"
        );
    }
}
