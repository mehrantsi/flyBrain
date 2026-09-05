use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ModelParameters {
    pub dt_ms: f64,
    pub resting_mv: f64,
    pub reset_mv: f64,
    pub threshold_mv: f64,
    pub membrane_tau_ms: f64,
    pub synapse_tau_ms: f64,
    pub refractory_ms: f64,
    pub delay_ms: f64,
    pub synapse_weight_mv: f64,
    #[serde(default = "default_external_weight_mv")]
    pub external_weight_mv: f64,
}

fn default_external_weight_mv() -> f64 {
    68.75
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            dt_ms: 0.1,
            resting_mv: -52.0,
            reset_mv: -52.0,
            threshold_mv: -45.0,
            membrane_tau_ms: 20.0,
            synapse_tau_ms: 5.0,
            refractory_ms: 2.2,
            delay_ms: 1.8,
            synapse_weight_mv: 0.275,
            external_weight_mv: 68.75,
        }
    }
}

impl ModelParameters {
    pub fn validate(self) -> Result<Self> {
        if !self.dt_ms.is_finite() || self.dt_ms <= 0.0 {
            bail!("dt_ms must be finite and positive");
        }
        for (name, value) in [
            ("membrane_tau_ms", self.membrane_tau_ms),
            ("synapse_tau_ms", self.synapse_tau_ms),
        ] {
            if !value.is_finite() || value <= 0.0 {
                bail!("{name} must be finite and positive");
            }
        }
        for (name, value) in [
            ("refractory_ms", self.refractory_ms),
            ("delay_ms", self.delay_ms),
        ] {
            if !value.is_finite() || value < 0.0 {
                bail!("{name} must be finite and non-negative");
            }
            self.exact_steps(value, name)?;
        }
        for (name, value) in [
            ("resting_mv", self.resting_mv),
            ("reset_mv", self.reset_mv),
            ("threshold_mv", self.threshold_mv),
            ("synapse_weight_mv", self.synapse_weight_mv),
            ("external_weight_mv", self.external_weight_mv),
        ] {
            if !value.is_finite() {
                bail!("{name} must be finite");
            }
        }
        Ok(self)
    }

    pub fn delay_steps(self) -> usize {
        (self.delay_ms / self.dt_ms).round() as usize
    }

    pub fn refractory_steps(self) -> i32 {
        (self.refractory_ms / self.dt_ms).round() as i32
    }

    pub fn membrane_decay(self) -> f64 {
        (-self.dt_ms / self.membrane_tau_ms).exp()
    }

    pub fn synapse_decay(self) -> f64 {
        (-self.dt_ms / self.synapse_tau_ms).exp()
    }

    pub fn coupling(self) -> f64 {
        if (self.membrane_tau_ms - self.synapse_tau_ms).abs() <= f64::EPSILON {
            self.dt_ms / self.membrane_tau_ms * self.membrane_decay()
        } else {
            self.synapse_tau_ms / (self.membrane_tau_ms - self.synapse_tau_ms)
                * (self.membrane_decay() - self.synapse_decay())
        }
    }

    fn exact_steps(self, duration_ms: f64, name: &str) -> Result<usize> {
        let steps = (duration_ms / self.dt_ms).round();
        if (steps * self.dt_ms - duration_ms).abs() > 1e-9 {
            bail!("{name} must be an integer multiple of dt_ms");
        }
        Ok(steps as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::ModelParameters;

    #[test]
    fn published_defaults_resolve_to_exact_steps() {
        let parameters = ModelParameters::default().validate().unwrap();

        assert_eq!(parameters.delay_steps(), 18);
        assert_eq!(parameters.refractory_steps(), 22);
        assert!((parameters.membrane_decay() - 0.9950124791926823).abs() < 1e-15);
        assert!((parameters.synapse_decay() - 0.9801986733067553).abs() < 1e-15);
    }
}
