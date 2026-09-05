use std::f64::consts::{PI, TAU};

use anyhow::{Result, bail};

pub const GROOMING_LEG_COUNT: usize = 6;
pub const GROOMING_CONTROL_COUNT: usize = 42;
pub const GROOMING_MIN_SUPPORT_LEGS: usize = 4;
pub const GROOMING_BOUT_DURATION_SECONDS: f64 = 1.8;
pub const GROOMING_FALLBACK_INTERVAL_SECONDS: f64 = 8.0;
const GROOMING_SUPPORT_LOSS_GRACE_SECONDS: f64 = 0.05;
const BRUSH_FREQUENCY_HZ: f64 = 8.0;
const JOINTS_PER_LEG: usize = 7;
#[allow(clippy::approx_constant)]
const JOINT_CONTROL_LIMIT_RAD: f64 = 3.14;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GroomingMode {
    #[default]
    None,
    AntennaBilateral,
    AntennaLeft,
    AntennaRight,
}

impl GroomingMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::AntennaBilateral => "ANTENNA-BOTH",
            Self::AntennaLeft => "ANTENNA-L",
            Self::AntennaRight => "ANTENNA-R",
        }
    }

    fn active_front_legs(self) -> [bool; 2] {
        match self {
            Self::None => [false, false],
            Self::AntennaBilateral => [true, true],
            Self::AntennaLeft => [true, false],
            Self::AntennaRight => [false, true],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GroomingTrigger {
    #[default]
    None,
    Manual,
    Fallback,
}

impl GroomingTrigger {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Manual => "manual",
            Self::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GroomingInput {
    pub dt_seconds: f64,
    pub grounded: bool,
    pub contact_count: usize,
    pub allow_fallback: bool,
    pub taste_active: bool,
    pub taste_valence: f64,
    pub feeding_extension: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GroomingCommand {
    pub mode: GroomingMode,
    pub trigger: GroomingTrigger,
    pub active: bool,
    pub phase: f64,
    pub support_leg_count: usize,
}

pub struct GroomingController {
    mode: GroomingMode,
    trigger: GroomingTrigger,
    elapsed_seconds: f64,
    support_loss_elapsed_seconds: f64,
    fallback_elapsed_seconds: f64,
    pending_manual: bool,
    waiting_for_support: bool,
    next_fallback_left: bool,
}

impl GroomingController {
    pub fn new() -> Self {
        Self {
            mode: GroomingMode::None,
            trigger: GroomingTrigger::None,
            elapsed_seconds: 0.0,
            support_loss_elapsed_seconds: 0.0,
            fallback_elapsed_seconds: 0.0,
            pending_manual: false,
            waiting_for_support: false,
            next_fallback_left: true,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn request_manual(&mut self) {
        self.pending_manual = true;
    }

    pub fn preparing(&self) -> bool {
        self.waiting_for_support
    }

    pub fn update(&mut self, input: GroomingInput) -> Result<GroomingCommand> {
        validate_input(input)?;
        let safe = input.grounded
            && !input.taste_active
            && input.taste_valence <= 0.0
            && input.feeding_extension <= 0.01;

        if self.active() {
            self.waiting_for_support = false;
            if !safe {
                self.abort();
                return Ok(self.command());
            }
            if input.contact_count < GROOMING_MIN_SUPPORT_LEGS {
                self.support_loss_elapsed_seconds += input.dt_seconds;
                if self.support_loss_elapsed_seconds >= GROOMING_SUPPORT_LOSS_GRACE_SECONDS {
                    self.abort();
                    return Ok(self.command());
                }
            } else {
                self.support_loss_elapsed_seconds = 0.0;
            }
            self.elapsed_seconds += input.dt_seconds;
            if self.elapsed_seconds >= GROOMING_BOUT_DURATION_SECONDS {
                self.finish_bout();
            }
        } else if safe {
            if input.allow_fallback {
                self.fallback_elapsed_seconds += input.dt_seconds;
            } else {
                self.fallback_elapsed_seconds = 0.0;
            }
            let bout_requested = self.pending_manual
                || (input.allow_fallback
                    && self.fallback_elapsed_seconds + 1e-9 >= GROOMING_FALLBACK_INTERVAL_SECONDS);
            self.waiting_for_support = bout_requested;
            if bout_requested && input.contact_count >= GROOMING_MIN_SUPPORT_LEGS {
                let manual = self.pending_manual;
                self.pending_manual = false;
                let (mode, trigger) = if manual {
                    (GroomingMode::AntennaBilateral, GroomingTrigger::Manual)
                } else {
                    let mode = if self.next_fallback_left {
                        GroomingMode::AntennaLeft
                    } else {
                        GroomingMode::AntennaRight
                    };
                    self.next_fallback_left = !self.next_fallback_left;
                    (mode, GroomingTrigger::Fallback)
                };
                self.start_bout(mode, trigger);
            }
        } else {
            self.fallback_elapsed_seconds = 0.0;
            self.waiting_for_support = false;
        }
        Ok(self.command())
    }

    pub fn apply(
        &self,
        joint_controls: &mut [f64; GROOMING_CONTROL_COUNT],
        adhesion: &mut [f64; GROOMING_LEG_COUNT],
    ) {
        if !self.active() {
            return;
        }

        let active_front_legs = self.mode.active_front_legs();
        for (leg_index, active) in active_front_legs.into_iter().enumerate() {
            if active {
                adhesion[leg_index * 3] = 0.0;
            }
        }
        for leg_index in [1, 2, 4, 5] {
            adhesion[leg_index] = 1.0;
        }

        let phase = self.phase();
        let envelope = (PI * phase).sin().max(0.0);
        let brush = (TAU * BRUSH_FREQUENCY_HZ * self.elapsed_seconds).sin();
        let lift = envelope * (0.72 + 0.28 * brush);
        for (front_index, active) in active_front_legs.into_iter().enumerate() {
            if !active {
                continue;
            }
            let leg_index = if front_index == 0 { 0 } else { 3 };
            let side = if front_index == 0 { -1.0 } else { 1.0 };
            let base = leg_index * JOINTS_PER_LEG;
            joint_controls[base] += side * 0.34 * envelope * brush;
            joint_controls[base + 1] += 0.30 * lift;
            joint_controls[base + 2] += side * 0.24 * envelope * brush;
            joint_controls[base + 3] -= 0.40 * lift;
            joint_controls[base + 4] -= side * 0.20 * envelope * brush;
            joint_controls[base + 5] -= 0.50 * lift;
            joint_controls[base + 6] -= 0.30 * lift;
            for control in &mut joint_controls[base..base + JOINTS_PER_LEG] {
                *control = control.clamp(-JOINT_CONTROL_LIMIT_RAD, JOINT_CONTROL_LIMIT_RAD);
            }
        }
    }

    fn active(&self) -> bool {
        self.mode != GroomingMode::None
    }

    fn phase(&self) -> f64 {
        (self.elapsed_seconds / GROOMING_BOUT_DURATION_SECONDS).clamp(0.0, 1.0)
    }

    fn command(&self) -> GroomingCommand {
        let active = self.active();
        let active_front_legs = self.mode.active_front_legs();
        let active_front_count = active_front_legs
            .into_iter()
            .filter(|active| *active)
            .count();
        GroomingCommand {
            mode: self.mode,
            trigger: self.trigger,
            active,
            phase: if active { self.phase() } else { 0.0 },
            support_leg_count: if active {
                GROOMING_LEG_COUNT - active_front_count
            } else {
                0
            },
        }
    }

    fn start_bout(&mut self, mode: GroomingMode, trigger: GroomingTrigger) {
        self.mode = mode;
        self.trigger = trigger;
        self.elapsed_seconds = 0.0;
        self.support_loss_elapsed_seconds = 0.0;
        self.fallback_elapsed_seconds = 0.0;
        self.waiting_for_support = false;
    }

    fn finish_bout(&mut self) {
        self.mode = GroomingMode::None;
        self.trigger = GroomingTrigger::None;
        self.elapsed_seconds = 0.0;
        self.support_loss_elapsed_seconds = 0.0;
        self.waiting_for_support = false;
    }

    fn abort(&mut self) {
        self.finish_bout();
        self.fallback_elapsed_seconds = 0.0;
    }
}

impl Default for GroomingController {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_input(input: GroomingInput) -> Result<()> {
    if !input.dt_seconds.is_finite()
        || input.dt_seconds <= 0.0
        || !(-1.0..=1.0).contains(&input.taste_valence)
        || !input.taste_valence.is_finite()
        || !input.feeding_extension.is_finite()
        || !(0.0..=1.0).contains(&input.feeding_extension)
    {
        bail!("grooming input is invalid")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(dt_seconds: f64) -> GroomingInput {
        GroomingInput {
            dt_seconds,
            grounded: true,
            contact_count: 6,
            ..GroomingInput::default()
        }
    }

    #[test]
    fn manual_bout_is_grounded_only_and_keeps_four_support_legs() {
        let mut controller = GroomingController::new();
        controller.request_manual();
        let command = controller.update(input(0.01)).unwrap();
        assert_eq!(command.mode, GroomingMode::AntennaBilateral);
        assert_eq!(command.trigger, GroomingTrigger::Manual);
        assert_eq!(command.support_leg_count, 4);

        let mut controls = [0.0; GROOMING_CONTROL_COUNT];
        let mut adhesion = [0.0; GROOMING_LEG_COUNT];
        controller.apply(&mut controls, &mut adhesion);
        assert_eq!(adhesion, [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        assert!(controls.iter().all(|control| control.is_finite()));
        assert!(
            controls.iter().all(
                |control| (-JOINT_CONTROL_LIMIT_RAD..=JOINT_CONTROL_LIMIT_RAD).contains(control)
            )
        );

        let airborne = GroomingInput {
            grounded: false,
            ..input(0.01)
        };
        assert!(!controller.update(airborne).unwrap().active);
    }

    #[test]
    fn taste_and_feeding_abort_a_bout() {
        let mut controller = GroomingController::new();
        controller.request_manual();
        assert!(controller.update(input(0.01)).unwrap().active);
        assert!(
            !controller
                .update(GroomingInput {
                    taste_active: true,
                    ..input(0.01)
                })
                .unwrap()
                .active
        );

        controller.request_manual();
        assert!(controller.update(input(0.01)).unwrap().active);
        assert!(
            !controller
                .update(GroomingInput {
                    feeding_extension: 0.5,
                    ..input(0.01)
                })
                .unwrap()
                .active
        );
    }

    #[test]
    fn unsafe_manual_request_waits_without_freezing_for_support() {
        let mut controller = GroomingController::new();
        controller.request_manual();
        let tasting = GroomingInput {
            taste_active: true,
            contact_count: 0,
            ..input(0.01)
        };
        assert!(!controller.update(tasting).unwrap().active);
        assert!(!controller.preparing());

        let unsupported = GroomingInput {
            contact_count: 2,
            ..input(0.01)
        };
        assert!(!controller.update(unsupported).unwrap().active);
        assert!(controller.preparing());

        let command = controller.update(input(0.01)).unwrap();
        assert!(command.active);
        assert_eq!(command.trigger, GroomingTrigger::Manual);
        assert!(!controller.preparing());
    }

    #[test]
    fn transient_support_loss_does_not_abort_a_bout() {
        let mut controller = GroomingController::new();
        controller.request_manual();
        assert!(controller.update(input(0.01)).unwrap().active);

        let unsupported = GroomingInput {
            contact_count: 3,
            ..input(0.01)
        };
        for _ in 0..4 {
            assert!(controller.update(unsupported).unwrap().active);
        }
        assert!(controller.update(input(0.01)).unwrap().active);
        for _ in 0..4 {
            assert!(controller.update(unsupported).unwrap().active);
        }
        assert!(!controller.update(unsupported).unwrap().active);
    }

    #[test]
    fn fallback_is_deterministic_and_alternates_sides() {
        let mut controller = GroomingController::new();
        let fallback_input = GroomingInput {
            allow_fallback: true,
            ..input(0.01)
        };
        for _ in 0..799 {
            let command = controller.update(fallback_input).unwrap();
            assert!(!command.active);
        }
        let mut command = controller.update(fallback_input).unwrap();
        assert_eq!(command.mode, GroomingMode::AntennaLeft);
        for _ in 0..180 {
            command = controller.update(fallback_input).unwrap();
        }
        assert!(!command.active);
        for _ in 0..800 {
            command = controller.update(fallback_input).unwrap();
        }
        assert_eq!(command.mode, GroomingMode::AntennaRight);
    }

    #[test]
    fn connected_brain_disables_idle_fallback_but_allows_manual_requests() {
        let mut controller = GroomingController::new();
        for _ in 0..801 {
            let command = controller.update(input(0.01)).unwrap();
            assert!(!command.active);
            assert_eq!(command.trigger, GroomingTrigger::None);
        }

        controller.request_manual();
        let command = controller.update(input(0.01)).unwrap();
        assert!(command.active);
        assert_eq!(command.trigger, GroomingTrigger::Manual);
    }

    #[test]
    fn left_and_right_sweeps_are_mirrored() {
        let mut controller = GroomingController::new();
        controller.request_manual();
        controller.update(input(0.01)).unwrap();
        controller.update(input(0.6)).unwrap();
        let mut bilateral = [0.0; GROOMING_CONTROL_COUNT];
        let mut adhesion = [0.0; GROOMING_LEG_COUNT];
        controller.apply(&mut bilateral, &mut adhesion);
        assert!((bilateral[0] + bilateral[21]).abs() < 1e-10);
        assert!((bilateral[2] + bilateral[23]).abs() < 1e-10);
        assert!((bilateral[1] - bilateral[22]).abs() < 1e-10);
        assert!(
            bilateral
                .iter()
                .all(|control| control.abs() <= JOINT_CONTROL_LIMIT_RAD)
        );
    }

    #[test]
    fn joint_overlay_respects_the_model_ctrlrange_literal() {
        let mut controller = GroomingController::new();
        controller.request_manual();
        controller.update(input(0.01)).unwrap();
        controller.update(input(0.6)).unwrap();
        let mut controls = [JOINT_CONTROL_LIMIT_RAD; GROOMING_CONTROL_COUNT];
        let mut adhesion = [0.0; GROOMING_LEG_COUNT];
        controller.apply(&mut controls, &mut adhesion);
        assert!(
            controls.iter().all(
                |control| (-JOINT_CONTROL_LIMIT_RAD..=JOINT_CONTROL_LIMIT_RAD).contains(control)
            )
        );
        assert!(
            controls
                .iter()
                .any(|control| (*control - JOINT_CONTROL_LIMIT_RAD).abs() < 1e-12)
        );
    }
}
