use std::f64::consts::PI;

pub const SAMPLE_RATE_HZ: f64 = 100.0;
pub const SAMPLE_PERIOD_SECONDS: f64 = 1.0 / SAMPLE_RATE_HZ;
pub const HIGH_PASS_CUTOFF_HZ: f64 = 1.0;
pub const LOW_PASS_CUTOFF_HZ: f64 = 30.0;
pub const SPECTRUM_WINDOW_SAMPLES: usize = 256;
pub const SPECTRUM_MIN_FREQUENCY_HZ: f64 = 1.0;
pub const SPECTRUM_MAX_FREQUENCY_HZ: f64 = LOW_PASS_CUTOFF_HZ;

const SIGNAL_POWER_EPSILON_MV2: f64 = 1.0e-14;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BrainSignalSample {
    pub filtered_field_mv: f64,
    pub dominant_frequency_hz: f64,
}

pub struct BrainSignalProcessor {
    previous_input_mv: f64,
    high_pass_mv: f64,
    filtered_field_mv: f64,
    samples: [f64; SPECTRUM_WINDOW_SAMPLES],
    sample_count: usize,
    write_index: usize,
    has_previous_input: bool,
    samples_since_spectrum: usize,
    dominant_frequency_hz: f64,
}

impl Default for BrainSignalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl BrainSignalProcessor {
    pub fn new() -> Self {
        Self {
            previous_input_mv: 0.0,
            high_pass_mv: 0.0,
            filtered_field_mv: 0.0,
            samples: [0.0; SPECTRUM_WINDOW_SAMPLES],
            sample_count: 0,
            write_index: 0,
            has_previous_input: false,
            samples_since_spectrum: 0,
            dominant_frequency_hz: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.previous_input_mv = 0.0;
        self.high_pass_mv = 0.0;
        self.filtered_field_mv = 0.0;
        self.samples.fill(0.0);
        self.sample_count = 0;
        self.write_index = 0;
        self.has_previous_input = false;
        self.samples_since_spectrum = 0;
        self.dominant_frequency_hz = 0.0;
    }

    pub fn update(&mut self, mean_membrane_deviation_mv: f64) -> BrainSignalSample {
        assert!(mean_membrane_deviation_mv.is_finite());
        let previous_input_mv = if self.has_previous_input {
            self.previous_input_mv
        } else {
            mean_membrane_deviation_mv
        };
        self.previous_input_mv = mean_membrane_deviation_mv;
        self.has_previous_input = true;

        let high_pass_alpha = high_pass_alpha();
        self.high_pass_mv =
            high_pass_alpha * (self.high_pass_mv + mean_membrane_deviation_mv - previous_input_mv);
        let low_pass_alpha = low_pass_alpha();
        self.filtered_field_mv += low_pass_alpha * (self.high_pass_mv - self.filtered_field_mv);

        self.samples[self.write_index] = self.filtered_field_mv;
        self.write_index = (self.write_index + 1) % SPECTRUM_WINDOW_SAMPLES;
        self.sample_count = (self.sample_count + 1).min(SPECTRUM_WINDOW_SAMPLES);
        self.samples_since_spectrum += 1;
        if self.sample_count >= 100 && self.samples_since_spectrum >= 25 {
            self.dominant_frequency_hz = self.compute_dominant_frequency_hz();
            self.samples_since_spectrum = 0;
        }

        BrainSignalSample {
            filtered_field_mv: self.filtered_field_mv,
            dominant_frequency_hz: self.dominant_frequency_hz,
        }
    }

    pub fn filtered_field_mv(&self) -> f64 {
        self.filtered_field_mv
    }

    pub fn dominant_frequency_hz(&self) -> f64 {
        self.dominant_frequency_hz
    }

    fn compute_dominant_frequency_hz(&self) -> f64 {
        let sample_count = self.sample_count;
        if sample_count < 2 {
            return 0.0;
        }

        let first_index =
            (self.write_index + SPECTRUM_WINDOW_SAMPLES - sample_count) % SPECTRUM_WINDOW_SAMPLES;
        let denominator = (sample_count - 1) as f64;
        let mut total_power = 0.0;
        let mut peak_power = SIGNAL_POWER_EPSILON_MV2;
        let mut peak_frequency_hz = 0.0;
        let minimum_bin =
            (SPECTRUM_MIN_FREQUENCY_HZ * sample_count as f64 / SAMPLE_RATE_HZ).ceil() as usize;
        let maximum_bin = ((SPECTRUM_MAX_FREQUENCY_HZ * sample_count as f64 / SAMPLE_RATE_HZ)
            .floor() as usize)
            .min(sample_count / 2);

        for bin in minimum_bin..=maximum_bin {
            let frequency_hz = bin as f64 * SAMPLE_RATE_HZ / sample_count as f64;
            let mut real = 0.0;
            let mut imaginary = 0.0;
            for sample_offset in 0..sample_count {
                let index = (first_index + sample_offset) % SPECTRUM_WINDOW_SAMPLES;
                let window = if sample_count == 1 {
                    1.0
                } else {
                    0.5 - 0.5 * (2.0 * PI * sample_offset as f64 / denominator).cos()
                };
                let phase = 2.0 * PI * bin as f64 * sample_offset as f64 / sample_count as f64;
                let value = self.samples[index] * window;
                real += value * phase.cos();
                imaginary -= value * phase.sin();
            }
            let power = real * real + imaginary * imaginary;
            total_power += power;
            if power > peak_power {
                peak_power = power;
                peak_frequency_hz = frequency_hz;
            }
        }

        if total_power <= SIGNAL_POWER_EPSILON_MV2 {
            0.0
        } else {
            peak_frequency_hz
        }
    }
}

fn high_pass_alpha() -> f64 {
    let tau_seconds = 1.0 / (2.0 * PI * HIGH_PASS_CUTOFF_HZ);
    tau_seconds / (tau_seconds + SAMPLE_PERIOD_SECONDS)
}

fn low_pass_alpha() -> f64 {
    let tau_seconds = 1.0 / (2.0 * PI * LOW_PASS_CUTOFF_HZ);
    SAMPLE_PERIOD_SECONDS / (tau_seconds + SAMPLE_PERIOD_SECONDS)
}

#[cfg(test)]
mod tests {
    use super::{BrainSignalProcessor, SAMPLE_RATE_HZ, SPECTRUM_WINDOW_SAMPLES};
    use std::f64::consts::PI;

    #[test]
    fn constant_input_is_rejected_by_the_high_pass_stage() {
        let mut processor = BrainSignalProcessor::new();
        let mut sample = processor.update(2.5);
        for _ in 1..SPECTRUM_WINDOW_SAMPLES + 32 {
            sample = processor.update(2.5);
        }

        assert!(sample.filtered_field_mv.abs() < 1.0e-12);
        assert_eq!(sample.dominant_frequency_hz, 0.0);
    }

    #[test]
    fn reset_discards_filter_and_spectrum_history() {
        let mut processor = BrainSignalProcessor::new();
        for index in 0..SPECTRUM_WINDOW_SAMPLES {
            processor.update((index as f64 * 0.4).sin());
        }
        assert_ne!(processor.filtered_field_mv(), 0.0);

        processor.reset();

        assert_eq!(processor.filtered_field_mv(), 0.0);
        assert_eq!(processor.dominant_frequency_hz(), 0.0);
        assert_eq!(processor.update(3.0).filtered_field_mv, 0.0);
    }

    #[test]
    fn dominant_frequency_tracks_an_eight_hz_sine() {
        let mut processor = BrainSignalProcessor::new();
        let mut sample = Default::default();
        for index in 0..SPECTRUM_WINDOW_SAMPLES {
            let time_seconds = index as f64 / SAMPLE_RATE_HZ;
            sample = processor.update((2.0 * PI * 8.0 * time_seconds).sin());
        }

        assert!((sample.dominant_frequency_hz - 8.0).abs() <= SAMPLE_RATE_HZ / 256.0);
    }

    #[test]
    fn zero_input_has_no_dominant_frequency() {
        let mut processor = BrainSignalProcessor::new();
        let mut sample = Default::default();
        for _ in 0..SPECTRUM_WINDOW_SAMPLES {
            sample = processor.update(0.0);
        }

        assert_eq!(sample.filtered_field_mv, 0.0);
        assert_eq!(sample.dominant_frequency_hz, 0.0);
    }

    #[test]
    #[should_panic]
    fn non_finite_input_is_rejected() {
        BrainSignalProcessor::new().update(f64::NAN);
    }
}
