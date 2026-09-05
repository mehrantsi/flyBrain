use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::cns_olfaction::CnsOlfactoryReadout;

const ACQUISITION_SECONDS: f64 = 0.5;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct OdorGuidanceParameters {
    pub enabled: bool,
    pub enter_rate_hz: f64,
    pub release_rate_hz: f64,
    pub steering_gain: f64,
    pub close_concentration_ppm: f64,
}

impl Default for OdorGuidanceParameters {
    fn default() -> Self {
        Self {
            enabled: true,
            enter_rate_hz: 10.0,
            release_rate_hz: 9.0,
            steering_gain: 150.0,
            close_concentration_ppm: 1.0,
        }
    }
}

impl OdorGuidanceParameters {
    pub fn validate(self) -> Result<Self> {
        if [
            self.enter_rate_hz,
            self.release_rate_hz,
            self.steering_gain,
            self.close_concentration_ppm,
        ]
        .into_iter()
        .any(|value| !value.is_finite() || value <= 0.0)
            || self.release_rate_hz >= self.enter_rate_hz
        {
            bail!("invalid neural odor guidance parameters")
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct OdorGuidanceCommand {
    pub active: bool,
    pub steering: f64,
    pub mean_rate_hz: f64,
    pub close: bool,
    pub approach_height_mm: f64,
}

#[derive(Default)]
pub struct OdorGuidance {
    active: bool,
    close: bool,
    filtered_contrast: f64,
    observation_seconds: f64,
    approach_height_mm: f64,
}

impl OdorGuidance {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn update(
        &mut self,
        readout: CnsOlfactoryReadout,
        height_mm: f64,
        dt_seconds: f64,
        eligible: bool,
        parameters: OdorGuidanceParameters,
    ) -> OdorGuidanceCommand {
        let rate = 0.5 * (readout.rate_hz[0] + readout.rate_hz[1]);
        if !eligible
            || !parameters.enabled
            || readout.observed_seconds < 0.2
            || rate < parameters.release_rate_hz
            || !rate.is_finite()
            || !dt_seconds.is_finite()
            || dt_seconds <= 0.0
        {
            self.reset();
            return OdorGuidanceCommand::default();
        }
        let concentration = 0.5 * (readout.concentration_ppm[0] + readout.concentration_ppm[1]);
        if !concentration.is_finite() {
            self.reset();
            return OdorGuidanceCommand::default();
        }
        let concentration = concentration.max(0.0);
        if !self.active && rate >= parameters.enter_rate_hz {
            self.active = true;
            self.approach_height_mm = height_mm.max(8.0);
        }
        if !self.active {
            return OdorGuidanceCommand::default();
        }
        self.observation_seconds += dt_seconds;
        if self.observation_seconds < ACQUISITION_SECONDS {
            return OdorGuidanceCommand {
                active: true,
                mean_rate_hz: rate,
                approach_height_mm: self.approach_height_mm,
                ..Default::default()
            };
        }
        let contrast = if concentration > 1.0 {
            (readout.concentration_ppm[0] - readout.concentration_ppm[1]) / (2.0 * concentration)
        } else {
            readout.contrast
        };
        self.filtered_contrast +=
            (1.0 - (-dt_seconds / 0.35).exp()) * (contrast - self.filtered_contrast);
        let normal_steering = (parameters.steering_gain * self.filtered_contrast).clamp(-0.7, 0.7);
        if concentration >= parameters.close_concentration_ppm {
            self.close = true;
        } else if concentration < parameters.close_concentration_ppm * 0.7 {
            self.close = false;
        }
        OdorGuidanceCommand {
            active: true,
            steering: normal_steering,
            mean_rate_hz: rate,
            close: self.close,
            approach_height_mm: self.approach_height_mm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn readout(left: f64, right: f64) -> CnsOlfactoryReadout {
        CnsOlfactoryReadout {
            observed_seconds: 1.0,
            rate_hz: [left, right],
            contrast: (left - right) / (left + right),
            spike_delta: 1,
            ..CnsOlfactoryReadout::default()
        }
    }

    #[test]
    fn baseline_and_disconnected_outputs_cannot_guide_motion() {
        let mut guidance = OdorGuidance::default();
        let parameters = OdorGuidanceParameters::default();
        assert!(
            !guidance
                .update(readout(8.0, 8.0), 2.0, 0.002, true, parameters)
                .active
        );
        assert!(
            !guidance
                .update(readout(40.0, 20.0), 2.0, 0.002, false, parameters)
                .active
        );
    }

    #[test]
    fn insufficient_observation_time_cannot_acquire_odor() {
        let signal = CnsOlfactoryReadout {
            observed_seconds: 0.199,
            ..readout(30.0, 20.0)
        };
        assert!(
            !OdorGuidance::default()
                .update(signal, 8.0, 0.2, true, OdorGuidanceParameters::default(),)
                .active
        );
    }

    #[test]
    fn eligibility_and_disabled_state_reset_guidance() {
        let mut guidance = OdorGuidance::default();
        let parameters = OdorGuidanceParameters::default();
        assert!(
            guidance
                .update(readout(20.0, 20.0), 20.0, 0.6, true, parameters)
                .active
        );

        let disabled = OdorGuidanceParameters {
            enabled: false,
            ..parameters
        };
        assert!(
            !guidance
                .update(readout(40.0, 20.0), 20.0, 0.2, true, disabled)
                .active
        );
        assert!(
            !guidance
                .update(readout(9.5, 9.5), 20.0, 0.2, true, parameters)
                .active
        );

        assert!(
            !guidance
                .update(readout(40.0, 20.0), 20.0, 0.2, false, parameters)
                .active
        );
        assert!(
            !guidance
                .update(readout(9.5, 9.5), 20.0, 0.2, true, parameters)
                .active
        );
    }

    #[test]
    fn enter_and_release_thresholds_have_hysteresis() {
        let mut guidance = OdorGuidance::default();
        let parameters = OdorGuidanceParameters::default();
        assert!(
            guidance
                .update(readout(10.0, 10.0), 20.0, 0.2, true, parameters)
                .active
        );
        assert!(
            guidance
                .update(readout(9.0, 9.0), 20.0, 0.2, true, parameters)
                .active
        );
        assert!(
            !guidance
                .update(readout(8.99, 8.99), 20.0, 0.2, true, parameters)
                .active
        );
    }

    #[test]
    fn acquisition_height_is_held_despite_transient_odor_or_height_changes() {
        let mut guidance = OdorGuidance::default();
        let parameters = OdorGuidanceParameters::default();
        let signal = CnsOlfactoryReadout {
            concentration_ppm: [2.0; 2],
            ..readout(20.0, 20.0)
        };
        assert_eq!(
            guidance
                .update(signal, 40.0, 0.6, true, parameters)
                .approach_height_mm,
            40.0
        );
        let adapted = CnsOlfactoryReadout {
            rate_hz: [14.0; 2],
            ..signal
        };
        assert_eq!(
            guidance
                .update(adapted, 30.0, 0.2, true, parameters)
                .approach_height_mm,
            40.0
        );
        let diluted = CnsOlfactoryReadout {
            concentration_ppm: [1.0; 2],
            ..adapted
        };
        let recovered = guidance.update(diluted, 30.0, 0.2, true, parameters);
        assert!(recovered.active);
        assert_eq!(recovered.approach_height_mm, 40.0);
    }

    #[test]
    fn swapped_neural_rates_reverse_bounded_guidance() {
        let parameters = OdorGuidanceParameters::default();
        let left = OdorGuidance::default().update(readout(20.0, 15.0), 2.0, 0.6, true, parameters);
        let right = OdorGuidance::default().update(readout(15.0, 20.0), 2.0, 0.6, true, parameters);
        assert!(left.steering > 0.0 && left.steering <= 0.7);
        assert_eq!(left.steering, -right.steering);
        assert_eq!(left.approach_height_mm, 8.0);
    }

    #[test]
    fn acquisition_waits_for_the_neural_filter_before_steering_or_landing() {
        let mut guidance = OdorGuidance::default();
        let parameters = OdorGuidanceParameters {
            close_concentration_ppm: 2.0,
            ..Default::default()
        };
        let signal = CnsOlfactoryReadout {
            concentration_ppm: [4.0, 3.0],
            ..readout(25.0, 20.0)
        };
        for _ in 0..4 {
            let command = guidance.update(signal, 8.0, 0.1, true, parameters);
            assert!(command.active);
            assert_eq!(command.steering, 0.0);
            assert!(!command.close);
        }
        let acquired = guidance.update(signal, 8.0, 0.1, true, parameters);
        assert!(acquired.steering > 0.0 && acquired.close);
    }

    #[test]
    fn directional_evidence_is_averaged_before_motor_saturation() {
        let mut guidance = OdorGuidance::default();
        let parameters = OdorGuidanceParameters::default();
        let mut mean_steering = 0.0;
        for step in 0..4000 {
            let contrast = if step % 2 == 0 { 0.1 } else { -0.099 };
            let command = guidance.update(
                CnsOlfactoryReadout {
                    contrast,
                    ..readout(20.0, 20.0)
                },
                8.0,
                0.002,
                true,
                parameters,
            );
            if step >= 3000 {
                mean_steering += command.steering / 1000.0;
            }
        }
        assert!((mean_steering - 0.075).abs() < 0.005, "{mean_steering}");
    }

    #[test]
    fn landing_context_uses_concentration_not_nonmonotonic_total_rate() {
        let parameters = OdorGuidanceParameters {
            close_concentration_ppm: 2.0,
            ..Default::default()
        };
        let mut guidance = OdorGuidance::default();
        let high_rate_far = CnsOlfactoryReadout {
            concentration_ppm: [1.0; 2],
            ..readout(90.0, 90.0)
        };
        assert!(
            !guidance
                .update(high_rate_far, 8.0, 0.6, true, parameters)
                .close
        );
        let adapted_close = CnsOlfactoryReadout {
            concentration_ppm: [4.0; 2],
            ..readout(40.0, 40.0)
        };
        assert!(
            guidance
                .update(adapted_close, 8.0, 0.1, true, parameters)
                .close
        );
        let between = CnsOlfactoryReadout {
            concentration_ppm: [1.5; 2],
            ..adapted_close
        };
        assert!(guidance.update(between, 8.0, 0.1, true, parameters).close);
        assert!(
            !guidance
                .update(high_rate_far, 8.0, 0.1, true, parameters)
                .close
        );
    }
}
