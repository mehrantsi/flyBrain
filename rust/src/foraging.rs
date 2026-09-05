use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::behavior::BehaviorMode;
use crate::flight_behavior::FlightMode;

const ODOR_APPROACH_ENTER: f64 = 0.035;
const ODOR_APPROACH_RELEASE: f64 = 0.012;
const ODOR_LANDING_THRESHOLD: f64 = 0.80;
const NEURAL_FOOD_LANDING_ODOR_THRESHOLD: f64 = 0.80;
const ODOR_LANDING_BILATERAL_FRACTION: f64 = 0.45;
const ODOR_LANDING_DWELL_SECONDS: f64 = 0.22;
const LANDING_DN_DWELL_SECONDS: f64 = 0.08;
const CNS_LANDING_INTEGRATION_SECONDS: f64 = 0.25;
const ODOR_LOSS_DWELL_SECONDS: f64 = 0.6;
const FOOD_LANDING_MAX_FLIGHT_DRIVE: f64 = 0.10;
const LANDING_DN_DRIVE_THRESHOLD: f64 = 0.02;
const APPROACH_HORIZONTAL_SPEED_SCALE: f64 = 0.24;
const DESCENT_HORIZONTAL_SPEED_SCALE: f64 = 0.10;
const TOUCHDOWN_STABILIZATION_SECONDS: f64 = 0.12;
const MINIMUM_GROUNDED_REST_SECONDS: f64 = 1.5;
const MAXIMUM_GROUND_SEARCH_SECONDS: f64 = 8.0;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct CnsForagingParameters {
    pub landing_odor_enter: f64,
    pub landing_odor_release: f64,
    pub landing_drive_enter: f64,
    pub landing_drive_release: f64,
    pub approach_speed_scale: f64,
}

impl Default for CnsForagingParameters {
    fn default() -> Self {
        Self {
            landing_odor_enter: 0.4,
            landing_odor_release: 0.2,
            landing_drive_enter: 0.02,
            landing_drive_release: 0.01,
            approach_speed_scale: 0.02,
        }
    }
}

impl CnsForagingParameters {
    pub fn validate(self) -> Result<Self> {
        if !self.landing_odor_enter.is_finite()
            || !(ODOR_APPROACH_ENTER..=2.0).contains(&self.landing_odor_enter)
            || !self.landing_odor_release.is_finite()
            || !(ODOR_APPROACH_RELEASE..self.landing_odor_enter)
                .contains(&self.landing_odor_release)
            || !self.landing_drive_enter.is_finite()
            || !(0.0..=1.0).contains(&self.landing_drive_enter)
            || !self.landing_drive_release.is_finite()
            || !(0.0..self.landing_drive_enter).contains(&self.landing_drive_release)
            || !self.approach_speed_scale.is_finite()
            || !(0.0..=1.0).contains(&self.approach_speed_scale)
        {
            bail!("CNS foraging calibration is invalid")
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ForagingMode {
    #[default]
    Search,
    Approach,
    Descend,
    GroundSearch,
    Feed,
    PostMeal,
    Rest,
}

impl ForagingMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Search => "SEARCH",
            Self::Approach => "APPROACH",
            Self::Descend => "DESCEND",
            Self::GroundSearch => "GROUND SEARCH",
            Self::Feed => "FEED",
            Self::PostMeal => "POST-MEAL",
            Self::Rest => "REST",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForagingInput {
    pub dt_seconds: f64,
    pub brain_enabled: bool,
    pub flight_mode: FlightMode,
    pub behavior_mode: BehaviorMode,
    pub odor_left: f64,
    pub odor_right: f64,
    pub taste_active: bool,
    pub surface_contact_count: usize,
    pub brain_flight_drive: f64,
    pub brain_landing_drive: f64,
    pub cns_calibration: Option<CnsForagingParameters>,
    pub cns_odor_guidance: Option<crate::odor_guidance::OdorGuidanceCommand>,
}

impl Default for ForagingInput {
    fn default() -> Self {
        Self {
            dt_seconds: 0.002,
            brain_enabled: false,
            flight_mode: FlightMode::Grounded,
            behavior_mode: BehaviorMode::Explore,
            odor_left: 0.0,
            odor_right: 0.0,
            taste_active: false,
            surface_contact_count: 0,
            brain_flight_drive: 0.0,
            brain_landing_drive: 0.0,
            cns_calibration: None,
            cns_odor_guidance: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForagingCommand {
    pub mode: ForagingMode,
    pub landing_request: bool,
    pub takeoff_inhibited: bool,
    pub horizontal_speed_scale: f64,
}

impl Default for ForagingCommand {
    fn default() -> Self {
        Self {
            mode: ForagingMode::Search,
            landing_request: false,
            takeoff_inhibited: false,
            horizontal_speed_scale: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ForagingController {
    mode: ForagingMode,
    landing_ready_seconds: f64,
    landing_drive_seconds: f64,
    cns_landing_evidence: f64,
    odor_lost_seconds: f64,
    grounded_seconds: f64,
    ground_search_exhausted: bool,
    cns_landing_consumed: bool,
    cns_odor_release_seconds: f64,
}

impl ForagingController {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn update(&mut self, input: ForagingInput) -> Result<ForagingCommand> {
        validate_input(input)?;
        let odor_total = input.odor_left + input.odor_right;
        let odor_balanced = odor_total > 0.0
            && (input.odor_left - input.odor_right).abs()
                <= ODOR_LANDING_BILATERAL_FRACTION * odor_total;
        let airborne = input.flight_mode != FlightMode::Grounded;
        let descent_speed_scale = if input.cns_odor_guidance.is_some() {
            0.0
        } else {
            DESCENT_HORIZONTAL_SPEED_SCALE
        };
        let landing_odor_threshold = input
            .cns_calibration
            .map_or(NEURAL_FOOD_LANDING_ODOR_THRESHOLD, |calibration| {
                calibration.landing_odor_enter
            });
        let landing_drive_threshold = input
            .cns_calibration
            .map_or(LANDING_DN_DRIVE_THRESHOLD, |calibration| {
                calibration.landing_drive_enter
            });
        if let Some(calibration) = input.cns_calibration {
            if input
                .cns_odor_guidance
                .map_or(odor_total < calibration.landing_odor_release, |guidance| {
                    !guidance.close
                })
            {
                self.cns_odor_release_seconds += input.dt_seconds;
                if self.cns_odor_release_seconds >= ODOR_LOSS_DWELL_SECONDS {
                    self.cns_landing_consumed = false;
                }
            } else {
                self.cns_odor_release_seconds = 0.0;
            }
        }
        let neural_landing_context = input.cns_odor_guidance.map_or(
            odor_total >= landing_odor_threshold && odor_balanced,
            |guidance| guidance.close,
        ) || (input.flight_mode == FlightMode::Cruise
            && input.surface_contact_count >= 2);
        let neural_landing_intent = input.brain_enabled
            && neural_landing_context
            && input.brain_landing_drive >= landing_drive_threshold;

        if input.behavior_mode == BehaviorMode::DepartFood {
            self.mode = ForagingMode::PostMeal;
            self.cns_landing_consumed = input.cns_calibration.is_some();
            self.landing_ready_seconds = 0.0;
            self.landing_drive_seconds = 0.0;
            self.cns_landing_evidence = 0.0;
            self.odor_lost_seconds = 0.0;
            return Ok(self.command(false, !airborne, 1.0));
        }
        if input.behavior_mode == BehaviorMode::Feed || input.taste_active {
            self.mode = ForagingMode::Feed;
            self.landing_ready_seconds = 0.0;
            self.landing_drive_seconds = 0.0;
            self.cns_landing_evidence = 0.0;
            self.odor_lost_seconds = 0.0;
            return Ok(self.command(airborne, !airborne, descent_speed_scale));
        }

        if self.ground_search_exhausted {
            if airborne || odor_total < ODOR_APPROACH_RELEASE {
                self.ground_search_exhausted = false;
            } else {
                return Ok(ForagingCommand::default());
            }
        }

        if self.mode == ForagingMode::PostMeal {
            self.mode = ForagingMode::Search;
            self.landing_ready_seconds = 0.0;
            self.landing_drive_seconds = 0.0;
            self.cns_landing_evidence = 0.0;
            self.odor_lost_seconds = 0.0;
            self.grounded_seconds = 0.0;
            self.ground_search_exhausted = !airborne && odor_total >= ODOR_APPROACH_RELEASE;
            return Ok(ForagingCommand::default());
        } else if self.mode == ForagingMode::Feed {
            self.mode = if airborne {
                ForagingMode::Search
            } else if odor_total >= ODOR_APPROACH_RELEASE {
                ForagingMode::GroundSearch
            } else {
                ForagingMode::Rest
            };
            self.landing_ready_seconds = 0.0;
            self.landing_drive_seconds = 0.0;
            self.cns_landing_evidence = 0.0;
            self.odor_lost_seconds = 0.0;
            self.grounded_seconds = 0.0;
            self.ground_search_exhausted = false;
        }

        if input.cns_calibration.is_some() {
            if airborne
                && input.brain_enabled
                && neural_landing_context
                && !self.cns_landing_consumed
            {
                let decay = (-input.dt_seconds / CNS_LANDING_INTEGRATION_SECONDS).exp();
                self.cns_landing_evidence = self.cns_landing_evidence * decay
                    + input.brain_landing_drive * CNS_LANDING_INTEGRATION_SECONDS * (1.0 - decay);
                if self.cns_landing_evidence >= landing_drive_threshold * LANDING_DN_DWELL_SECONDS {
                    self.mode = ForagingMode::Descend;
                    self.cns_landing_consumed = true;
                }
            } else {
                self.cns_landing_evidence = 0.0;
            }
        } else if airborne && neural_landing_intent && !self.cns_landing_consumed {
            self.landing_drive_seconds += input.dt_seconds;
            if self.landing_drive_seconds >= LANDING_DN_DWELL_SECONDS {
                self.mode = ForagingMode::Descend;
                self.cns_landing_consumed = input.cns_calibration.is_some();
            }
        } else {
            self.landing_drive_seconds = 0.0;
        }
        if self.mode == ForagingMode::Descend {
            if airborne {
                return Ok(self.command(true, false, descent_speed_scale));
            }
            self.mode = if odor_total >= ODOR_APPROACH_RELEASE {
                ForagingMode::GroundSearch
            } else {
                ForagingMode::Rest
            };
            self.grounded_seconds = 0.0;
        }

        if matches!(self.mode, ForagingMode::GroundSearch | ForagingMode::Rest) {
            if airborne {
                self.mode = ForagingMode::Search;
                self.grounded_seconds = 0.0;
            } else {
                self.grounded_seconds += input.dt_seconds;
                if input.brain_enabled {
                    let continuing_ground_search =
                        input
                            .cns_calibration
                            .map_or(neural_landing_intent, |calibration| {
                                self.mode == ForagingMode::GroundSearch
                                    && input.cns_odor_guidance.map_or(
                                        odor_total >= calibration.landing_odor_release
                                            && input.brain_landing_drive
                                                >= calibration.landing_drive_release,
                                        |guidance| guidance.close,
                                    )
                            });
                    if self.grounded_seconds < TOUCHDOWN_STABILIZATION_SECONDS
                        || continuing_ground_search
                    {
                        return Ok(self.command(false, true, 1.0));
                    }
                    self.mode = ForagingMode::Search;
                    self.grounded_seconds = 0.0;
                    self.ground_search_exhausted = false;
                    return Ok(ForagingCommand::default());
                }
                let searching = self.mode == ForagingMode::GroundSearch
                    && odor_total >= ODOR_APPROACH_RELEASE
                    && self.grounded_seconds + 1e-9 < MAXIMUM_GROUND_SEARCH_SECONDS;
                if self.grounded_seconds < MINIMUM_GROUNDED_REST_SECONDS || searching {
                    return Ok(self.command(false, true, 1.0));
                }
                self.ground_search_exhausted = self.mode == ForagingMode::GroundSearch;
                self.mode = ForagingMode::Search;
                self.grounded_seconds = 0.0;
                if self.ground_search_exhausted {
                    return Ok(ForagingCommand::default());
                }
            }
        }

        let odor_engaged = input
            .cns_odor_guidance
            .map_or(odor_total >= ODOR_APPROACH_ENTER, |guidance| {
                guidance.active
            })
            && (!input.brain_enabled
                || input.cns_calibration.is_some()
                || input.brain_flight_drive <= FOOD_LANDING_MAX_FLIGHT_DRIVE);
        if self.mode == ForagingMode::Search && odor_engaged {
            self.mode = ForagingMode::Approach;
            self.landing_ready_seconds = 0.0;
            self.odor_lost_seconds = 0.0;
            self.grounded_seconds = 0.0;
        }
        if self.mode == ForagingMode::Approach {
            if input
                .cns_odor_guidance
                .map_or(odor_total < ODOR_APPROACH_RELEASE, |guidance| {
                    !guidance.active
                })
            {
                self.odor_lost_seconds += input.dt_seconds;
                if self.odor_lost_seconds >= ODOR_LOSS_DWELL_SECONDS {
                    self.mode = ForagingMode::Search;
                    self.landing_ready_seconds = 0.0;
                    self.odor_lost_seconds = 0.0;
                    return Ok(ForagingCommand::default());
                }
            } else {
                self.odor_lost_seconds = 0.0;
            }
            if !airborne {
                self.grounded_seconds += input.dt_seconds;
                if input.brain_enabled {
                    return Ok(self.command(
                        false,
                        input.cns_calibration.is_none() && neural_landing_intent,
                        1.0,
                    ));
                }
                if self.grounded_seconds + 1e-9 >= MAXIMUM_GROUND_SEARCH_SECONDS {
                    self.mode = ForagingMode::Search;
                    self.grounded_seconds = 0.0;
                    self.ground_search_exhausted = true;
                    return Ok(ForagingCommand::default());
                }
                return Ok(self.command(false, true, 1.0));
            }
            if !input.brain_enabled
                && odor_total >= ODOR_LANDING_THRESHOLD
                && odor_balanced
                && input.brain_flight_drive <= FOOD_LANDING_MAX_FLIGHT_DRIVE
            {
                self.landing_ready_seconds += input.dt_seconds;
                if self.landing_ready_seconds >= ODOR_LANDING_DWELL_SECONDS {
                    self.mode = ForagingMode::Descend;
                    return Ok(self.command(true, false, DESCENT_HORIZONTAL_SPEED_SCALE));
                }
            } else {
                self.landing_ready_seconds = 0.0;
            }
            return Ok(self.command(
                false,
                false,
                if input
                    .cns_odor_guidance
                    .is_some_and(|guidance| guidance.close)
                {
                    0.0
                } else {
                    input
                        .cns_calibration
                        .map_or(APPROACH_HORIZONTAL_SPEED_SCALE, |calibration| {
                            calibration.approach_speed_scale
                        })
                },
            ));
        }

        Ok(ForagingCommand::default())
    }

    fn command(
        self,
        landing_request: bool,
        takeoff_inhibited: bool,
        horizontal_speed_scale: f64,
    ) -> ForagingCommand {
        ForagingCommand {
            mode: self.mode,
            landing_request,
            takeoff_inhibited,
            horizontal_speed_scale,
        }
    }
}

fn validate_input(input: ForagingInput) -> Result<()> {
    if let Some(calibration) = input.cns_calibration {
        calibration.validate()?;
    }
    if !input.dt_seconds.is_finite()
        || input.dt_seconds <= 0.0
        || [
            input.odor_left,
            input.odor_right,
            input.brain_flight_drive,
            input.brain_landing_drive,
        ]
        .into_iter()
        .any(|value| !value.is_finite() || value < 0.0)
        || input.brain_flight_drive > 1.0
        || input.brain_landing_drive > 1.0
    {
        bail!("foraging input is invalid")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brief_taste_retains_close_odor_ground_search_without_takeoff() {
        let mut controller = ForagingController::default();
        let mut input = ForagingInput {
            brain_enabled: true,
            flight_mode: FlightMode::Grounded,
            behavior_mode: BehaviorMode::Feed,
            taste_active: true,
            odor_left: 0.7,
            odor_right: 0.7,
            brain_flight_drive: 0.9,
            cns_calibration: Some(CnsForagingParameters::default()),
            cns_odor_guidance: Some(crate::odor_guidance::OdorGuidanceCommand {
                active: true,
                close: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(controller.update(input).unwrap().mode, ForagingMode::Feed);
        input.behavior_mode = BehaviorMode::TrackOdor;
        input.taste_active = false;
        for _ in 0..500 {
            let command = controller.update(input).unwrap();
            assert_eq!(command.mode, ForagingMode::GroundSearch);
            assert!(command.takeoff_inhibited);
        }
    }

    #[test]
    fn neural_odor_landing_needs_both_close_context_and_landing_output() {
        let mut controller = ForagingController::default();
        let mut input = ForagingInput {
            brain_enabled: true,
            flight_mode: FlightMode::Cruise,
            odor_left: 1.0,
            odor_right: 1.0,
            brain_flight_drive: 0.9,
            brain_landing_drive: 0.04,
            cns_calibration: Some(CnsForagingParameters::default()),
            cns_odor_guidance: Some(crate::odor_guidance::OdorGuidanceCommand {
                active: true,
                close: false,
                ..Default::default()
            }),
            ..ForagingInput::default()
        };
        for _ in 0..100 {
            assert!(!controller.update(input).unwrap().landing_request);
        }
        input.cns_odor_guidance.as_mut().unwrap().close = true;
        input.brain_landing_drive = 0.0;
        for _ in 0..100 {
            assert!(!controller.update(input).unwrap().landing_request);
        }
        input.brain_landing_drive = 0.04;
        for _ in 0..100 {
            controller.update(input).unwrap();
        }
        assert!(controller.update(input).unwrap().landing_request);
        input.flight_mode = FlightMode::Grounded;
        input.brain_landing_drive = 0.0;
        for _ in 0..100 {
            assert!(controller.update(input).unwrap().takeoff_inhibited);
        }
        input.cns_odor_guidance.as_mut().unwrap().close = false;
        assert!(!controller.update(input).unwrap().takeoff_inhibited);
    }

    fn airborne() -> ForagingInput {
        ForagingInput {
            brain_enabled: true,
            flight_mode: FlightMode::Cruise,
            brain_flight_drive: 0.04,
            ..ForagingInput::default()
        }
    }

    fn cns_airborne() -> ForagingInput {
        ForagingInput {
            brain_enabled: true,
            flight_mode: FlightMode::Cruise,
            brain_flight_drive: 0.9,
            cns_calibration: Some(CnsForagingParameters::default()),
            ..ForagingInput::default()
        }
    }

    #[test]
    fn centered_odor_does_not_request_landing_without_landing_dn_activity() {
        let mut controller = ForagingController::default();
        let input = ForagingInput {
            odor_left: 0.03,
            odor_right: 0.03,
            ..airborne()
        };
        let approach = controller.update(input).unwrap();
        assert_eq!(approach.mode, ForagingMode::Approach);
        assert_eq!(
            approach.horizontal_speed_scale,
            APPROACH_HORIZONTAL_SPEED_SCALE
        );
        let mut command = approach;
        for _ in 0..120 {
            command = controller.update(input).unwrap();
        }
        assert_eq!(command.mode, ForagingMode::Approach);
        assert!(!command.landing_request);
        assert_eq!(
            command.horizontal_speed_scale,
            APPROACH_HORIZONTAL_SPEED_SCALE
        );
    }

    #[test]
    fn departing_food_takes_priority_over_stale_taste() {
        let mut controller = ForagingController::default();
        let command = controller
            .update(ForagingInput {
                behavior_mode: BehaviorMode::DepartFood,
                taste_active: true,
                flight_mode: FlightMode::Grounded,
                ..ForagingInput::default()
            })
            .unwrap();
        assert_eq!(command.mode, ForagingMode::PostMeal);
        assert!(command.takeoff_inhibited);
        assert!(!command.landing_request);
    }

    #[test]
    fn cleared_feed_enters_deterministic_ground_rest() {
        let mut controller = ForagingController {
            mode: ForagingMode::Feed,
            ..ForagingController::default()
        };
        let input = ForagingInput {
            dt_seconds: 0.01,
            flight_mode: FlightMode::Grounded,
            ..ForagingInput::default()
        };
        let command = controller.update(input).unwrap();
        assert_eq!(command.mode, ForagingMode::Rest);
        assert!(command.takeoff_inhibited);

        for _ in 0..148 {
            let command = controller.update(input).unwrap();
            assert_eq!(command.mode, ForagingMode::Rest);
            assert!(command.takeoff_inhibited);
        }
        let released = controller.update(input).unwrap();
        assert_eq!(released.mode, ForagingMode::Search);
        assert!(!released.takeoff_inhibited);
    }

    #[test]
    fn completed_post_meal_departure_releases_neural_takeoff_despite_odor() {
        let mut controller = ForagingController {
            mode: ForagingMode::PostMeal,
            ..ForagingController::default()
        };
        let input = ForagingInput {
            flight_mode: FlightMode::Grounded,
            odor_left: ODOR_APPROACH_RELEASE,
            odor_right: ODOR_APPROACH_RELEASE,
            ..ForagingInput::default()
        };
        let command = controller.update(input).unwrap();
        assert_eq!(command.mode, ForagingMode::Search);
        assert!(!command.takeoff_inhibited);
        let held_clear = controller.update(input).unwrap();
        assert_eq!(held_clear.mode, ForagingMode::Search);
        assert!(!held_clear.takeoff_inhibited);
    }

    #[test]
    fn strong_flight_drive_vetoes_odor_landing() {
        let mut controller = ForagingController::default();
        for _ in 0..500 {
            let command = controller
                .update(ForagingInput {
                    odor_left: 0.08,
                    odor_right: 0.08,
                    brain_flight_drive: 0.5,
                    ..airborne()
                })
                .unwrap();
            assert_eq!(command.mode, ForagingMode::Search);
            assert!(!command.landing_request);
        }
    }

    #[test]
    fn no_brain_centered_odor_still_requests_landing() {
        let mut controller = ForagingController::default();
        let input = ForagingInput {
            dt_seconds: 0.05,
            flight_mode: FlightMode::Cruise,
            odor_left: 0.5,
            odor_right: 0.5,
            ..ForagingInput::default()
        };
        let mut command = controller.update(input).unwrap();
        assert_eq!(command.mode, ForagingMode::Approach);
        for _ in 0..5 {
            command = controller.update(input).unwrap();
        }
        assert_eq!(command.mode, ForagingMode::Descend);
        assert!(command.landing_request);
    }

    #[test]
    fn landing_dn_activity_needs_food_or_surface_context() {
        let mut controller = ForagingController::default();
        let input = ForagingInput {
            dt_seconds: 0.02,
            brain_landing_drive: LANDING_DN_DRIVE_THRESHOLD,
            ..airborne()
        };
        for _ in 0..20 {
            let command = controller.update(input).unwrap();
            assert_eq!(command.mode, ForagingMode::Search);
            assert!(!command.landing_request);
        }

        let contextual = ForagingInput {
            odor_left: NEURAL_FOOD_LANDING_ODOR_THRESHOLD * 0.5,
            odor_right: NEURAL_FOOD_LANDING_ODOR_THRESHOLD * 0.5,
            ..input
        };
        let mut command = controller.update(contextual).unwrap();
        for _ in 0..3 {
            command = controller.update(contextual).unwrap();
        }
        assert_eq!(command.mode, ForagingMode::Descend);
        assert!(command.landing_request);
    }

    #[test]
    fn weak_room_odor_does_not_turn_landing_activity_into_takeoff_inhibition() {
        let mut controller = ForagingController::default();
        let command = controller
            .update(ForagingInput {
                brain_enabled: true,
                flight_mode: FlightMode::Grounded,
                odor_left: 0.16,
                odor_right: 0.16,
                brain_landing_drive: LANDING_DN_DRIVE_THRESHOLD,
                brain_flight_drive: 0.04,
                ..ForagingInput::default()
            })
            .unwrap();
        assert_eq!(command.mode, ForagingMode::Approach);
        assert!(!command.takeoff_inhibited);
    }

    #[test]
    fn ordinary_ground_support_does_not_become_a_permanent_landing_command() {
        let mut controller = ForagingController::default();
        let command = controller
            .update(ForagingInput {
                brain_enabled: true,
                flight_mode: FlightMode::Grounded,
                surface_contact_count: 4,
                brain_landing_drive: LANDING_DN_DRIVE_THRESHOLD,
                brain_flight_drive: 0.04,
                ..ForagingInput::default()
            })
            .unwrap();
        assert!(!command.takeoff_inhibited);
    }

    #[test]
    fn landing_dn_activity_can_request_a_contact_surface_landing_without_odor() {
        let mut controller = ForagingController::default();
        let input = ForagingInput {
            dt_seconds: 0.02,
            brain_landing_drive: LANDING_DN_DRIVE_THRESHOLD,
            surface_contact_count: 2,
            ..airborne()
        };
        let mut command = controller.update(input).unwrap();
        for _ in 0..3 {
            command = controller.update(input).unwrap();
        }
        assert_eq!(command.mode, ForagingMode::Descend);
        assert!(command.landing_request);
    }

    #[test]
    fn brain_touchdown_has_only_a_short_stabilization_dwell() {
        let mut controller = ForagingController {
            mode: ForagingMode::Descend,
            ..ForagingController::default()
        };
        let grounded = ForagingInput {
            dt_seconds: 0.01,
            brain_enabled: true,
            flight_mode: FlightMode::Grounded,
            odor_left: 0.03,
            odor_right: 0.03,
            brain_flight_drive: 0.04,
            ..ForagingInput::default()
        };
        let command = controller.update(grounded).unwrap();
        assert_eq!(command.mode, ForagingMode::GroundSearch);
        assert!(command.takeoff_inhibited);
        for _ in 0..12 {
            controller.update(grounded).unwrap();
        }
        let released = controller.update(grounded).unwrap();
        assert!(!released.takeoff_inhibited);
    }

    #[test]
    fn brain_touchdown_without_odor_has_only_a_short_stabilization_dwell() {
        let mut controller = ForagingController {
            mode: ForagingMode::Descend,
            ..ForagingController::default()
        };
        let grounded = ForagingInput {
            dt_seconds: 0.01,
            brain_enabled: true,
            flight_mode: FlightMode::Grounded,
            brain_flight_drive: 0.04,
            ..ForagingInput::default()
        };
        let command = controller.update(grounded).unwrap();
        assert_eq!(command.mode, ForagingMode::Rest);
        assert!(command.takeoff_inhibited);
        for _ in 0..12 {
            controller.update(grounded).unwrap();
        }
        let released = controller.update(grounded).unwrap();
        assert!(!released.takeoff_inhibited);
    }

    #[test]
    fn airborne_taste_requests_a_controlled_descent() {
        let mut controller = ForagingController::default();
        let command = controller
            .update(ForagingInput {
                taste_active: true,
                behavior_mode: BehaviorMode::Feed,
                ..airborne()
            })
            .unwrap();
        assert_eq!(command.mode, ForagingMode::Feed);
        assert!(command.landing_request);
        assert!(!command.takeoff_inhibited);
    }

    #[test]
    fn cns_wing_activation_does_not_veto_odor_approach() {
        let mut controller = ForagingController::default();
        let command = controller
            .update(ForagingInput {
                odor_left: 0.1,
                odor_right: 0.1,
                ..cns_airborne()
            })
            .unwrap();
        assert_eq!(command.mode, ForagingMode::Approach);
        assert!(!command.landing_request);
        assert_eq!(
            command.horizontal_speed_scale,
            CnsForagingParameters::default().approach_speed_scale
        );
    }

    #[test]
    fn cns_diffuse_initial_odor_does_not_request_landing() {
        let mut controller = ForagingController::default();
        let input = ForagingInput {
            dt_seconds: 0.01,
            odor_left: 0.16,
            odor_right: 0.16,
            brain_landing_drive: 0.04,
            ..cns_airborne()
        };
        for _ in 0..20 {
            let command = controller.update(input).unwrap();
            assert_eq!(command.mode, ForagingMode::Approach);
            assert!(!command.landing_request);
        }
    }

    #[test]
    fn cns_balanced_odor_and_landing_drive_integrate_to_one_descent() {
        let mut controller = ForagingController::default();
        let input = ForagingInput {
            dt_seconds: 0.02,
            odor_left: 0.25,
            odor_right: 0.25,
            brain_landing_drive: 0.025,
            ..cns_airborne()
        };
        for _ in 0..3 {
            let command = controller.update(input).unwrap();
            assert_eq!(command.mode, ForagingMode::Approach);
            assert!(!command.landing_request);
        }
        let command = controller.update(input).unwrap();
        assert_eq!(command.mode, ForagingMode::Descend);
        assert!(command.landing_request);
        for _ in 0..20 {
            let command = controller.update(input).unwrap();
            assert_eq!(command.mode, ForagingMode::Descend);
            assert!(command.landing_request);
        }
    }

    #[test]
    fn cns_disconnected_landing_drive_cannot_request_descent() {
        let mut controller = ForagingController::default();
        let input = ForagingInput {
            dt_seconds: 0.02,
            odor_left: 0.25,
            odor_right: 0.25,
            brain_landing_drive: 0.0,
            ..cns_airborne()
        };
        for _ in 0..20 {
            let command = controller.update(input).unwrap();
            assert_eq!(command.mode, ForagingMode::Approach);
            assert!(!command.landing_request);
        }
    }

    #[test]
    fn cns_brief_odor_and_landing_burst_does_not_request_descent() {
        let mut controller = ForagingController::default();
        let burst = ForagingInput {
            dt_seconds: 0.02,
            odor_left: 0.25,
            odor_right: 0.25,
            brain_landing_drive: 0.04,
            ..cns_airborne()
        };
        for _ in 0..1 {
            let command = controller.update(burst).unwrap();
            assert_ne!(command.mode, ForagingMode::Descend);
            assert!(!command.landing_request);
        }
        let cleared = ForagingInput {
            odor_left: 0.0,
            odor_right: 0.0,
            brain_landing_drive: 0.0,
            ..burst
        };
        for _ in 0..20 {
            let command = controller.update(cleared).unwrap();
            assert_ne!(command.mode, ForagingMode::Descend);
            assert!(!command.landing_request);
        }
    }

    #[test]
    fn cns_landing_integrates_sparse_bursts_and_clears_on_context_loss() {
        let mut controller = ForagingController::default();
        let input = ForagingInput {
            dt_seconds: 0.01,
            odor_left: 0.25,
            odor_right: 0.25,
            brain_landing_drive: 0.08,
            ..cns_airborne()
        };
        for index in 0..7 {
            let command = controller
                .update(ForagingInput {
                    brain_landing_drive: if index % 3 == 0 { 0.08 } else { 0.0 },
                    ..input
                })
                .unwrap();
            assert_eq!(command.landing_request, index == 6);
        }
        controller.reset();
        controller.update(input).unwrap();
        assert!(controller.cns_landing_evidence > 0.0);
        controller
            .update(ForagingInput {
                odor_left: 0.0,
                odor_right: 0.0,
                ..input
            })
            .unwrap();
        assert_eq!(controller.cns_landing_evidence, 0.0);
        assert!(!controller.update(input).unwrap().landing_request);
    }

    #[test]
    fn cns_close_odor_brakes_approach_without_inventing_landing_activity() {
        let mut controller = ForagingController::default();
        for _ in 0..500 {
            let command = controller
                .update(ForagingInput {
                    cns_odor_guidance: Some(crate::odor_guidance::OdorGuidanceCommand {
                        active: true,
                        close: true,
                        ..Default::default()
                    }),
                    brain_landing_drive: 0.0,
                    ..cns_airborne()
                })
                .unwrap();
            assert_eq!(command.mode, ForagingMode::Approach);
            assert_eq!(command.horizontal_speed_scale, 0.0);
            assert!(!command.landing_request);
        }
    }

    #[test]
    fn cns_landing_rearms_in_diffuse_background_without_leaving_approach() {
        let mut controller = ForagingController {
            mode: ForagingMode::Approach,
            cns_landing_consumed: true,
            ..ForagingController::default()
        };
        let diffuse = ForagingInput {
            dt_seconds: 0.02,
            odor_left: 0.03,
            odor_right: 0.03,
            brain_landing_drive: 0.04,
            ..cns_airborne()
        };
        for _ in 0..29 {
            controller.update(diffuse).unwrap();
            assert!(controller.cns_landing_consumed);
        }
        assert_eq!(
            controller.update(diffuse).unwrap().mode,
            ForagingMode::Approach
        );
        assert!(!controller.cns_landing_consumed);
        for _ in 0..4 {
            controller
                .update(ForagingInput {
                    odor_left: 0.25,
                    odor_right: 0.25,
                    ..diffuse
                })
                .unwrap();
        }
        assert_eq!(controller.mode, ForagingMode::Descend);
    }

    #[test]
    fn cns_consumed_landing_does_not_repeat_until_odor_is_lost() {
        let mut controller = ForagingController::default();
        let landing = ForagingInput {
            dt_seconds: 0.02,
            odor_left: 0.25,
            odor_right: 0.25,
            brain_landing_drive: 0.04,
            ..cns_airborne()
        };
        for _ in 0..4 {
            controller.update(landing).unwrap();
        }
        let grounded = ForagingInput {
            flight_mode: FlightMode::Grounded,
            ..landing
        };
        let command = controller.update(grounded).unwrap();
        assert_eq!(command.mode, ForagingMode::GroundSearch);
        assert!(command.takeoff_inhibited);
        for _ in 0..20 {
            let command = controller.update(grounded).unwrap();
            assert_eq!(command.mode, ForagingMode::GroundSearch);
            assert!(command.takeoff_inhibited);
        }

        let airborne_again = controller.update(landing).unwrap();
        assert_eq!(airborne_again.mode, ForagingMode::Approach);
        assert!(!airborne_again.landing_request);
        for _ in 0..10 {
            let command = controller.update(landing).unwrap();
            assert_eq!(command.mode, ForagingMode::Approach);
            assert!(!command.landing_request);
        }

        let odor_lost = ForagingInput {
            odor_left: 0.0,
            odor_right: 0.0,
            brain_landing_drive: 0.04,
            ..landing
        };
        for _ in 0..29 {
            let command = controller.update(odor_lost).unwrap();
            assert_eq!(command.mode, ForagingMode::Approach);
            assert!(!command.landing_request);
        }
        let released = controller.update(odor_lost).unwrap();
        assert_eq!(released.mode, ForagingMode::Search);
        assert!(!released.landing_request);

        let mut command = controller.update(landing).unwrap();
        for _ in 0..3 {
            command = controller.update(landing).unwrap();
        }
        assert_eq!(command.mode, ForagingMode::Descend);
        assert!(command.landing_request);
    }

    #[test]
    fn cns_postmeal_release_allows_airborne_search_without_relanding() {
        let mut controller = ForagingController::default();
        let departure = ForagingInput {
            behavior_mode: BehaviorMode::DepartFood,
            flight_mode: FlightMode::Grounded,
            odor_left: 0.25,
            odor_right: 0.25,
            brain_flight_drive: 0.9,
            ..cns_airborne()
        };
        let command = controller.update(departure).unwrap();
        assert_eq!(command.mode, ForagingMode::PostMeal);
        assert!(command.takeoff_inhibited);

        let released = controller
            .update(ForagingInput {
                behavior_mode: BehaviorMode::Explore,
                flight_mode: FlightMode::Cruise,
                ..departure
            })
            .unwrap();
        assert_eq!(released.mode, ForagingMode::Search);
        assert!(!released.takeoff_inhibited);
        assert!(!released.landing_request);

        let approach = controller
            .update(ForagingInput {
                behavior_mode: BehaviorMode::Explore,
                flight_mode: FlightMode::Cruise,
                ..departure
            })
            .unwrap();
        assert_eq!(approach.mode, ForagingMode::Approach);
        assert!(!approach.takeoff_inhibited);
        assert!(!approach.landing_request);
    }

    #[test]
    fn cns_calibration_requires_valid_hysteresis() {
        let valid = CnsForagingParameters::default();
        assert!(
            ForagingController::default()
                .update(ForagingInput {
                    cns_calibration: Some(valid),
                    ..ForagingInput::default()
                })
                .is_ok()
        );
        let invalid = [
            CnsForagingParameters {
                landing_odor_enter: ODOR_APPROACH_ENTER - 1e-6,
                ..valid
            },
            CnsForagingParameters {
                landing_odor_release: valid.landing_odor_enter,
                ..valid
            },
            CnsForagingParameters {
                landing_odor_release: ODOR_APPROACH_RELEASE - 1e-6,
                ..valid
            },
            CnsForagingParameters {
                landing_drive_enter: 1.1,
                ..valid
            },
            CnsForagingParameters {
                landing_drive_release: valid.landing_drive_enter,
                ..valid
            },
            CnsForagingParameters {
                landing_drive_release: -1e-6,
                ..valid
            },
            CnsForagingParameters {
                approach_speed_scale: 1.1,
                ..valid
            },
            CnsForagingParameters {
                landing_odor_enter: f64::NAN,
                ..valid
            },
        ];
        for (index, calibration) in invalid.into_iter().enumerate() {
            assert!(
                ForagingController::default()
                    .update(ForagingInput {
                        cns_calibration: Some(calibration),
                        ..ForagingInput::default()
                    })
                    .is_err(),
                "invalid calibration {index} was accepted: {calibration:?}"
            );
        }
    }
}
