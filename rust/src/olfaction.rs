use anyhow::{Result, bail};

pub const ANTENNA_COUNT: usize = 2;
pub const FOOD_ODOR_BAND_COUNT: usize = 4;
pub const ORN_SPONTANEOUS_RATE_HZ: f64 = 8.0;
pub const ORN_MAXIMUM_RATE_HZ: f64 = 200.0;

const RECEPTOR_HALF_MAX_PPM: [f64; FOOD_ODOR_BAND_COUNT] = [1.0, 3.0, 12.0, 32.0];
const HILL_EXPONENT: f64 = 1.42;
const FAST_ADAPTATION_REFERENCE_PPM: f64 = 3.0;
const FAST_ADAPTATION_TAU_SECONDS: f64 = 0.25;
const RESPONSE_RISE_TAU_SECONDS: f64 = 0.04;
const RESPONSE_FALL_TAU_SECONDS: f64 = 0.20;
const BEHAVIORAL_ADAPTATION_TAU_SECONDS: f64 = 9.8;
const BEHAVIORAL_HALF_SATURATION_PPM: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum AntennaSide {
    Left = 0,
    Right = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum FoodOdorBand {
    Attractive = 0,
    Core = 1,
    HighConcentration = 2,
    AversiveHighConcentration = 3,
}

impl FoodOdorBand {
    pub const ALL: [Self; FOOD_ODOR_BAND_COUNT] = [
        Self::Attractive,
        Self::Core,
        Self::HighConcentration,
        Self::AversiveHighConcentration,
    ];

    pub fn from_profile_name(name: &str) -> Result<Self> {
        match name {
            "attractive" => Ok(Self::Attractive),
            "core" => Ok(Self::Core),
            "high_concentration" => Ok(Self::HighConcentration),
            "aversive_high_concentration" => Ok(Self::AversiveHighConcentration),
            _ => bail!("unsupported food-odor response band {name:?}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OlfactorySample {
    pub concentration_ppm: [f64; ANTENNA_COUNT],
    pub receptor_activation: [[f64; FOOD_ODOR_BAND_COUNT]; ANTENNA_COUNT],
    pub perceived_intensity: [f64; ANTENNA_COUNT],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct OlfactoryTransducer {
    fast_background_ppm: [f64; ANTENNA_COUNT],
    behavioral_background_ppm: [f64; ANTENNA_COUNT],
    receptor_activation: [[f64; FOOD_ODOR_BAND_COUNT]; ANTENNA_COUNT],
}

impl OlfactoryTransducer {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn update(
        &mut self,
        concentration_ppm: [f64; ANTENNA_COUNT],
        dt_seconds: f64,
    ) -> Result<OlfactorySample> {
        if !dt_seconds.is_finite() || dt_seconds <= 0.0 {
            bail!("olfactory timestep must be finite and positive")
        }
        if concentration_ppm
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            bail!("olfactory concentrations must be finite and non-negative")
        }

        let fast_adaptation_alpha = 1.0 - (-dt_seconds / FAST_ADAPTATION_TAU_SECONDS).exp();
        let behavioral_adaptation_alpha =
            1.0 - (-dt_seconds / BEHAVIORAL_ADAPTATION_TAU_SECONDS).exp();
        for (side, &concentration) in concentration_ppm.iter().enumerate() {
            let concentration_power = concentration.powf(HILL_EXPONENT);
            let gain = 1.0 / (1.0 + self.fast_background_ppm[side] / FAST_ADAPTATION_REFERENCE_PPM);
            for band in FoodOdorBand::ALL {
                let half_max = RECEPTOR_HALF_MAX_PPM[band as usize];
                let target = gain * concentration_power
                    / (concentration_power + half_max.powf(HILL_EXPONENT));
                let current = &mut self.receptor_activation[side][band as usize];
                let tau = if target >= *current {
                    RESPONSE_RISE_TAU_SECONDS
                } else {
                    RESPONSE_FALL_TAU_SECONDS
                };
                let alpha = 1.0 - (-dt_seconds / tau).exp();
                *current += alpha * (target - *current);
            }
            self.fast_background_ppm[side] +=
                fast_adaptation_alpha * (concentration - self.fast_background_ppm[side]);
            self.behavioral_background_ppm[side] += behavioral_adaptation_alpha
                * (concentration - self.behavioral_background_ppm[side]);
        }

        let perceived_intensity = std::array::from_fn(|side| {
            let concentration = concentration_ppm[side];
            concentration
                / (concentration
                    + BEHAVIORAL_HALF_SATURATION_PPM
                    + self.behavioral_background_ppm[side])
        });
        Ok(OlfactorySample {
            concentration_ppm,
            receptor_activation: self.receptor_activation,
            perceived_intensity,
        })
    }

    pub fn preview(&self, concentration_ppm: [f64; ANTENNA_COUNT]) -> Result<OlfactorySample> {
        if concentration_ppm
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            bail!("olfactory concentrations must be finite and non-negative")
        }
        let perceived_intensity = std::array::from_fn(|side| {
            let concentration = concentration_ppm[side];
            concentration
                / (concentration
                    + BEHAVIORAL_HALF_SATURATION_PPM
                    + self.behavioral_background_ppm[side])
        });
        Ok(OlfactorySample {
            concentration_ppm,
            receptor_activation: self.receptor_activation,
            perceived_intensity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settle(
        transducer: &mut OlfactoryTransducer,
        concentration_ppm: [f64; 2],
        seconds: f64,
    ) -> OlfactorySample {
        let dt = 0.002;
        let mut sample = OlfactorySample::default();
        for _ in 0..(seconds / dt) as usize {
            sample = transducer.update(concentration_ppm, dt).unwrap();
        }
        sample
    }

    #[test]
    fn zero_concentration_has_no_evoked_response() {
        let sample = settle(&mut OlfactoryTransducer::default(), [0.0; 2], 0.1);
        assert_eq!(sample.receptor_activation, [[0.0; 4]; 2]);
        assert_eq!(sample.perceived_intensity, [0.0; 2]);
    }

    #[test]
    fn behaviorally_relevant_food_odor_recruits_low_threshold_channels_first() {
        let sample = settle(&mut OlfactoryTransducer::default(), [3.0; 2], 0.1);
        let left = sample.receptor_activation[AntennaSide::Left as usize];
        assert!(left[FoodOdorBand::Attractive as usize] > 0.35);
        assert!(left[FoodOdorBand::Core as usize] > 0.2);
        assert!(left[FoodOdorBand::HighConcentration as usize] < left[FoodOdorBand::Core as usize]);
        assert!(
            left[FoodOdorBand::AversiveHighConcentration as usize]
                < left[FoodOdorBand::HighConcentration as usize]
        );
    }

    #[test]
    fn high_concentration_recruits_the_aversive_band() {
        let sample = settle(&mut OlfactoryTransducer::default(), [32.0; 2], 0.1);
        assert!(
            sample.receptor_activation[0][FoodOdorBand::AversiveHighConcentration as usize] > 0.1
        );
    }

    #[test]
    fn sustained_background_adapts_but_does_not_erase_food_odor() {
        let mut transducer = OlfactoryTransducer::default();
        let peak = settle(&mut transducer, [3.0; 2], 0.1).perceived_intensity[0];
        let adapted = settle(&mut transducer, [3.0; 2], 4.0).perceived_intensity[0];
        assert!(adapted < peak);
        assert!(adapted > 0.1);
    }

    #[test]
    fn bilateral_concentration_difference_is_preserved() {
        let sample = settle(&mut OlfactoryTransducer::default(), [3.0, 0.3], 0.1);
        assert!(sample.perceived_intensity[0] > sample.perceived_intensity[1]);
    }

    #[test]
    fn response_recovers_after_odor_is_removed() {
        let mut transducer = OlfactoryTransducer::default();
        settle(&mut transducer, [12.0; 2], 2.0);
        let residual = settle(&mut transducer, [0.0; 2], 0.2);
        assert!(residual.receptor_activation[0][0] > 0.0);
        let recovered = settle(&mut transducer, [0.0; 2], 2.0);
        assert!(recovered.receptor_activation[0][0] < 1e-3);
    }

    #[test]
    fn reset_removes_adaptation_and_receptor_state() {
        let mut transducer = OlfactoryTransducer::default();
        settle(&mut transducer, [12.0; 2], 2.0);
        transducer.reset();
        assert_eq!(transducer, OlfactoryTransducer::default());
    }

    #[test]
    fn preview_is_side_effect_free() {
        let mut transducer = OlfactoryTransducer::default();
        settle(&mut transducer, [3.0; 2], 0.5);
        let before = transducer.clone();
        let sample = transducer.preview([12.0, 0.0]).unwrap();
        assert_eq!(transducer, before);
        assert!(sample.perceived_intensity[0] > sample.perceived_intensity[1]);
    }
}
