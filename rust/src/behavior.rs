use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

const FEEDING_BOUT_SECONDS: f64 = 3.0;
const POST_MEAL_REFRACTORY_SECONDS: f64 = 12.0;
const POST_MEAL_DEPARTURE_SECONDS: f64 = 1.25;
const DEPARTURE_FORWARD_GAIN: f64 = 0.9;
const DEPARTURE_TURN_GAIN: f64 = 0.35;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct BehaviorParameters {
    pub feeding_bout_seconds: f64,
    pub post_meal_refractory_seconds: f64,
    #[serde(default = "default_post_meal_departure_seconds")]
    pub post_meal_departure_seconds: f64,
    pub departure_forward_gain: f64,
    pub departure_turn_gain: f64,
    pub wander_tau_seconds: f64,
    pub odor_tracking_threshold: f64,
    pub odor_confidence_scale: f64,
    pub explore_forward_gain: f64,
    pub explore_wander_gain: f64,
    pub odor_forward_bias: f64,
    pub odor_forward_gain: f64,
    pub odor_turn_gain: f64,
    #[serde(default = "default_odor_gradient_gain")]
    pub odor_gradient_gain: f64,
    pub odor_wander_gain: f64,
}

fn default_odor_gradient_gain() -> f64 {
    100.0
}

fn default_post_meal_departure_seconds() -> f64 {
    POST_MEAL_DEPARTURE_SECONDS
}

impl Default for BehaviorParameters {
    fn default() -> Self {
        Self {
            feeding_bout_seconds: FEEDING_BOUT_SECONDS,
            post_meal_refractory_seconds: POST_MEAL_REFRACTORY_SECONDS,
            post_meal_departure_seconds: POST_MEAL_DEPARTURE_SECONDS,
            departure_forward_gain: DEPARTURE_FORWARD_GAIN,
            departure_turn_gain: DEPARTURE_TURN_GAIN,
            wander_tau_seconds: 0.8,
            odor_tracking_threshold: 0.025,
            odor_confidence_scale: 0.3,
            explore_forward_gain: 0.72,
            explore_wander_gain: 0.62,
            odor_forward_bias: 0.65,
            odor_forward_gain: 0.35,
            odor_turn_gain: 0.92,
            odor_gradient_gain: default_odor_gradient_gain(),
            odor_wander_gain: 0.18,
        }
    }
}

impl BehaviorParameters {
    pub fn validate(self) -> Result<Self> {
        if [
            self.feeding_bout_seconds,
            self.post_meal_refractory_seconds,
            self.post_meal_departure_seconds,
            self.wander_tau_seconds,
            self.odor_tracking_threshold,
            self.odor_confidence_scale,
            self.odor_gradient_gain,
        ]
        .into_iter()
        .any(|value| !value.is_finite() || value <= 0.0)
            || [
                self.departure_forward_gain,
                self.departure_turn_gain,
                self.explore_forward_gain,
                self.explore_wander_gain,
                self.odor_forward_bias,
                self.odor_forward_gain,
                self.odor_turn_gain,
                self.odor_wander_gain,
            ]
            .into_iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            bail!("behavior parameters are invalid")
        }
        if self.post_meal_departure_seconds > self.post_meal_refractory_seconds {
            bail!("post-meal departure cannot outlast taste refractory")
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BehaviorMode {
    #[default]
    Explore,
    TrackOdor,
    Feed,
    DepartFood,
}

impl BehaviorMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Explore => "EXPLORE",
            Self::TrackOdor => "ODOR TAXIS",
            Self::Feed => "FEED",
            Self::DepartFood => "POST-MEAL",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BehaviorInput {
    pub dt_seconds: f64,
    pub odor_left: f64,
    pub odor_right: f64,
    pub taste_valence: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BehaviorCommand {
    pub mode: BehaviorMode,
    pub forward_gain: f64,
    pub turn_gain: f64,
    pub sensory_taste_gain: f64,
}

pub struct ExplorerController {
    parameters: BehaviorParameters,
    random_state: u64,
    wander: f64,
    feeding_elapsed_seconds: f64,
    taste_refractory_seconds: f64,
    leaving_food: bool,
    departure_elapsed_seconds: f64,
    departure_turn_gain: f64,
}

impl ExplorerController {
    pub fn new(seed: u64) -> Self {
        Self::with_parameters(seed, BehaviorParameters::default())
            .expect("default behavior parameters are valid")
    }

    pub fn with_parameters(seed: u64, parameters: BehaviorParameters) -> Result<Self> {
        Ok(Self {
            parameters: parameters.validate()?,
            random_state: seed,
            wander: 0.0,
            feeding_elapsed_seconds: 0.0,
            taste_refractory_seconds: 0.0,
            leaving_food: false,
            departure_elapsed_seconds: 0.0,
            departure_turn_gain: 0.0,
        })
    }

    pub fn reset(&mut self, seed: u64) {
        let parameters = self.parameters;
        *self = Self::with_parameters(seed, parameters)
            .expect("existing behavior parameters are valid");
    }

    pub fn parameters(&self) -> BehaviorParameters {
        self.parameters
    }

    pub fn update(&mut self, input: BehaviorInput) -> Result<BehaviorCommand> {
        if !input.dt_seconds.is_finite()
            || input.dt_seconds <= 0.0
            || !input.odor_left.is_finite()
            || input.odor_left < 0.0
            || !input.odor_right.is_finite()
            || input.odor_right < 0.0
            || !input.taste_valence.is_finite()
            || !(-1.0..=1.0).contains(&input.taste_valence)
        {
            bail!("behavior input is invalid")
        }

        let random = self.next_unit() * 2.0 - 1.0;
        let alpha = 1.0 - (-input.dt_seconds / self.parameters.wander_tau_seconds).exp();
        self.wander += alpha * (random - self.wander);
        self.taste_refractory_seconds = (self.taste_refractory_seconds - input.dt_seconds).max(0.0);

        if self.leaving_food {
            self.departure_elapsed_seconds += input.dt_seconds;
            if self.departure_elapsed_seconds < self.parameters.post_meal_departure_seconds {
                return Ok(self.departure_command());
            }
            self.leaving_food = false;
        }

        if input.taste_valence > 0.0 && self.taste_refractory_seconds == 0.0 {
            self.feeding_elapsed_seconds += input.dt_seconds;
            if self.feeding_elapsed_seconds >= self.parameters.feeding_bout_seconds {
                self.feeding_elapsed_seconds = 0.0;
                self.taste_refractory_seconds = self.parameters.post_meal_refractory_seconds;
                self.begin_departure(random);
                return Ok(self.departure_command());
            }
            return Ok(BehaviorCommand {
                mode: BehaviorMode::Feed,
                forward_gain: 0.0,
                turn_gain: 0.0,
                sensory_taste_gain: 1.0,
            });
        }
        self.feeding_elapsed_seconds = 0.0;

        if self.taste_refractory_seconds > 0.0 {
            return Ok(BehaviorCommand {
                mode: BehaviorMode::Explore,
                forward_gain: self.parameters.explore_forward_gain,
                turn_gain: (self.parameters.explore_wander_gain * self.wander).clamp(-0.7, 0.7),
                sensory_taste_gain: 0.0,
            });
        }

        let odor_total = input.odor_left + input.odor_right;
        if odor_total > self.parameters.odor_tracking_threshold {
            let bilateral = (self.parameters.odor_gradient_gain
                * (input.odor_left - input.odor_right)
                / (odor_total + 1e-9))
                .clamp(-1.0, 1.0);
            let confidence = (odor_total / self.parameters.odor_confidence_scale).clamp(0.0, 1.0);
            return Ok(BehaviorCommand {
                mode: BehaviorMode::TrackOdor,
                forward_gain: self.parameters.odor_forward_bias
                    + self.parameters.odor_forward_gain * confidence,
                turn_gain: (self.parameters.odor_turn_gain * bilateral
                    + self.parameters.odor_wander_gain * self.wander)
                    .clamp(-1.0, 1.0),
                sensory_taste_gain: 1.0,
            });
        }

        Ok(BehaviorCommand {
            mode: BehaviorMode::Explore,
            forward_gain: self.parameters.explore_forward_gain,
            turn_gain: (self.parameters.explore_wander_gain * self.wander).clamp(-0.7, 0.7),
            sensory_taste_gain: 1.0,
        })
    }

    fn begin_departure(&mut self, random: f64) {
        self.leaving_food = true;
        self.departure_elapsed_seconds = 0.0;
        self.departure_turn_gain = if random.is_sign_negative() {
            -self.parameters.departure_turn_gain
        } else {
            self.parameters.departure_turn_gain
        };
    }

    fn departure_command(&self) -> BehaviorCommand {
        let turn_scale = (1.0
            - self.departure_elapsed_seconds / self.parameters.post_meal_departure_seconds)
            .clamp(0.0, 1.0);
        BehaviorCommand {
            mode: BehaviorMode::DepartFood,
            forward_gain: self.parameters.departure_forward_gain,
            turn_gain: self.departure_turn_gain * turn_scale,
            sensory_taste_gain: 0.0,
        }
    }

    fn next_unit(&mut self) -> f64 {
        self.random_state = self
            .random_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        f64::from((self.random_state >> 32) as u32) / f64::from(u32::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taste_stops_locomotion() {
        let mut controller = ExplorerController::new(7);
        let command = controller
            .update(BehaviorInput {
                dt_seconds: 0.002,
                taste_valence: 0.8,
                ..BehaviorInput::default()
            })
            .unwrap();
        assert_eq!(command.mode, BehaviorMode::Feed);
        assert_eq!(command.forward_gain, 0.0);
        assert_eq!(command.turn_gain, 0.0);
        assert_eq!(command.sensory_taste_gain, 1.0);
    }

    #[test]
    fn completed_meal_adapts_taste_and_releases_locomotion() {
        let mut controller = ExplorerController::new(7);
        let input = BehaviorInput {
            dt_seconds: 0.01,
            taste_valence: 1.0,
            ..BehaviorInput::default()
        };
        let mut command = controller.update(input).unwrap();
        while command.mode == BehaviorMode::Feed {
            command = controller.update(input).unwrap();
        }
        assert_eq!(command.mode, BehaviorMode::DepartFood);
        assert!(command.forward_gain > 0.0);
        assert_eq!(command.sensory_taste_gain, 0.0);

        let mut departure_updates = 0;
        while command.mode == BehaviorMode::DepartFood {
            departure_updates += 1;
            command = controller.update(input).unwrap();
        }
        assert!(departure_updates < 200);
        assert_eq!(command.mode, BehaviorMode::Explore);
        assert_eq!(command.sensory_taste_gain, 0.0);

        for _ in 0..500 {
            command = controller.update(input).unwrap();
            assert_ne!(command.mode, BehaviorMode::Feed);
            assert_eq!(command.sensory_taste_gain, 0.0);
        }

        let clear = controller
            .update(BehaviorInput {
                dt_seconds: 0.01,
                ..BehaviorInput::default()
            })
            .unwrap();
        assert_eq!(clear.mode, BehaviorMode::Explore);
        assert!(clear.forward_gain > 0.0);
    }

    #[test]
    fn bilateral_odor_drives_turning_without_target_coordinates() {
        let mut controller = ExplorerController::new(11);
        let left = controller
            .update(BehaviorInput {
                dt_seconds: 0.002,
                odor_left: 0.4,
                odor_right: 0.05,
                ..BehaviorInput::default()
            })
            .unwrap();
        assert_eq!(left.mode, BehaviorMode::TrackOdor);
        assert!(left.turn_gain > 0.0);
        assert!(left.forward_gain > 0.65);
    }

    #[test]
    fn antennal_gradient_gain_resolves_a_small_plume_difference() {
        let mut controller = ExplorerController::new(13);
        let command = controller
            .update(BehaviorInput {
                dt_seconds: 0.002,
                odor_left: 0.0174,
                odor_right: 0.0172,
                ..BehaviorInput::default()
            })
            .unwrap();
        assert!(command.turn_gain > 0.1);
    }

    #[test]
    fn exploration_is_seed_deterministic() {
        let mut left = ExplorerController::new(19);
        let mut right = ExplorerController::new(19);
        for _ in 0..100 {
            let input = BehaviorInput {
                dt_seconds: 0.002,
                ..BehaviorInput::default()
            };
            assert_eq!(left.update(input).unwrap(), right.update(input).unwrap());
        }
    }
}
