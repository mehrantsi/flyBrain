use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

pub const GAIT_LEGS: [&str; 6] = ["lf", "lm", "lh", "rf", "rm", "rh"];
pub const GAIT_JOINTS_PER_LEG: usize = 7;
pub const GAIT_CONTROL_COUNT: usize = 42;
pub const GAIT_JOINT_ORDER: [&str; GAIT_JOINTS_PER_LEG] = [
    "coxa_yaw",
    "coxa_pitch",
    "coxa_roll",
    "femur_pitch",
    "femur_roll",
    "tibia_pitch",
    "tarsus_pitch",
];

#[derive(Clone, Debug, Deserialize)]
pub struct GaitLibrary {
    pub schema: String,
    pub source: String,
    pub runtime_interpolation: String,
    pub legs: Vec<String>,
    pub joint_order: Vec<String>,
    pub sample_count: usize,
    pub cycle_frequency_hz: f64,
    pub tripod_phase_offsets_rad: Vec<f64>,
    pub neutral_joint_angles: Vec<f64>,
    pub joint_angles: BTreeMap<String, Vec<[f64; GAIT_JOINTS_PER_LEG]>>,
    pub adhesion: BTreeMap<String, Vec<bool>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GaitCommand {
    pub joint_controls: [f64; GAIT_CONTROL_COUNT],
    pub adhesion: [f64; 6],
}

impl GaitLibrary {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let gait: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        gait.validate()?;
        Ok(gait)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != "flybrain-gait-v1" {
            bail!("unsupported gait schema: {}", self.schema)
        }
        if self.runtime_interpolation != "cyclic-linear" {
            bail!(
                "unsupported gait interpolation: {}",
                self.runtime_interpolation
            )
        }
        if self.legs.iter().map(String::as_str).ne(GAIT_LEGS) {
            bail!("gait leg order must be lf,lm,lh,rf,rm,rh")
        }
        if self
            .joint_order
            .iter()
            .map(String::as_str)
            .ne(GAIT_JOINT_ORDER)
        {
            bail!("gait joint order does not match the NeuroMechFly actuator order")
        }
        if self.sample_count < 16 {
            bail!("gait requires at least 16 phase samples")
        }
        if !self.cycle_frequency_hz.is_finite() || self.cycle_frequency_hz <= 0.0 {
            bail!("gait cycle frequency must be finite and positive")
        }
        if self.tripod_phase_offsets_rad.len() != GAIT_LEGS.len()
            || self
                .tripod_phase_offsets_rad
                .iter()
                .any(|value| !value.is_finite())
        {
            bail!("gait tripod phase offsets are invalid")
        }
        if self.neutral_joint_angles.len() != GAIT_CONTROL_COUNT
            || self
                .neutral_joint_angles
                .iter()
                .any(|value| !value.is_finite())
        {
            bail!("gait neutral controls are invalid")
        }
        for leg in GAIT_LEGS {
            let samples = self
                .joint_angles
                .get(leg)
                .with_context(|| format!("gait is missing {leg} joint samples"))?;
            if samples.len() != self.sample_count
                || samples.iter().flatten().any(|value| !value.is_finite())
            {
                bail!("gait joint samples are invalid for {leg}")
            }
            let adhesion = self
                .adhesion
                .get(leg)
                .with_context(|| format!("gait is missing {leg} adhesion samples"))?;
            if adhesion.len() != self.sample_count {
                bail!("gait adhesion samples are invalid for {leg}")
            }
        }
        Ok(())
    }

    pub fn sample(&self, phase_rad: f64, forward_gain: f64, turn_gain: f64) -> Result<GaitCommand> {
        if !phase_rad.is_finite() || !forward_gain.is_finite() || !turn_gain.is_finite() {
            bail!("gait phase and gains must be finite")
        }
        let forward_gain = forward_gain.clamp(0.0, 2.0);
        let turn_gain = turn_gain.clamp(-1.0, 1.0);
        self.sample_bilateral(
            phase_rad,
            [
                forward_gain * (1.0 - 0.5 * turn_gain),
                forward_gain * (1.0 + 0.5 * turn_gain),
            ],
        )
    }

    pub fn sample_bilateral(&self, phase_rad: f64, side_drive: [f64; 2]) -> Result<GaitCommand> {
        if !phase_rad.is_finite() || side_drive.iter().any(|drive| !drive.is_finite()) {
            bail!("gait phase and gains must be finite")
        }
        let mut joint_controls = [0.0; GAIT_CONTROL_COUNT];
        let mut adhesion = [0.0; 6];
        for (leg_index, leg) in GAIT_LEGS.iter().enumerate() {
            let side = usize::from(leg_index >= 3);
            let drive = side_drive[side].clamp(-2.0, 2.0);
            let phase_direction = if drive < 0.0 { -1.0 } else { 1.0 };
            let phase = (phase_direction * phase_rad + self.tripod_phase_offsets_rad[leg_index])
                .rem_euclid(TAU);
            let coordinate = phase / TAU * self.sample_count as f64;
            let lower = coordinate.floor() as usize % self.sample_count;
            let upper = (lower + 1) % self.sample_count;
            let fraction = coordinate - coordinate.floor();
            let leg_gain = drive.abs();
            let samples = &self.joint_angles[*leg];
            for (joint_index, (&lower_value, &upper_value)) in
                samples[lower].iter().zip(&samples[upper]).enumerate()
            {
                let offset = leg_index * GAIT_JOINTS_PER_LEG + joint_index;
                let interpolated = lower_value * (1.0 - fraction) + upper_value * fraction;
                let neutral = self.neutral_joint_angles[offset];
                joint_controls[offset] = neutral + leg_gain * (interpolated - neutral);
            }
            adhesion[leg_index] = f64::from(self.adhesion[*leg][lower]);
        }
        Ok(GaitCommand {
            joint_controls,
            adhesion,
        })
    }

    pub fn advance_phase(&self, phase_rad: f64, dt_seconds: f64, speed_gain: f64) -> Result<f64> {
        if !phase_rad.is_finite() || !dt_seconds.is_finite() || dt_seconds <= 0.0 {
            bail!("gait phase and timestep must be finite and timestep positive")
        }
        if !speed_gain.is_finite() || speed_gain < 0.0 {
            bail!("gait speed gain must be finite and non-negative")
        }
        Ok((phase_rad + TAU * self.cycle_frequency_hz * speed_gain * dt_seconds).rem_euclid(TAU))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exported_gait_is_cyclic_and_finite() {
        let gait = GaitLibrary::open("assets/neuromechfly/tripod_gait.json").unwrap();
        let at_zero = gait.sample(0.0, 1.0, 0.0).unwrap();
        let at_cycle = gait.sample(TAU, 1.0, 0.0).unwrap();
        assert_eq!(at_zero, at_cycle);
        assert!(at_zero.joint_controls.iter().all(|value| value.is_finite()));
        let mid_swing = gait.sample(0.5, 1.0, 0.0).unwrap();
        assert!(mid_swing.adhesion.contains(&0.0));
        assert!(mid_swing.adhesion.contains(&1.0));
    }

    #[test]
    fn zero_gain_returns_neutral_controls() {
        let gait = GaitLibrary::open("assets/neuromechfly/tripod_gait.json").unwrap();
        let command = gait.sample(1.25, 0.0, 0.0).unwrap();
        assert_eq!(command.joint_controls.as_slice(), gait.neutral_joint_angles);
    }

    #[test]
    fn legacy_sample_delegates_to_bilateral_drive_mapping() {
        let gait = GaitLibrary::open("assets/neuromechfly/tripod_gait.json").unwrap();
        let legacy = gait.sample(0.7, 1.8, -0.9).unwrap();
        let bilateral = gait
            .sample_bilateral(0.7, [1.8 * (1.0 + 0.45), 1.8 * (1.0 - 0.45)])
            .unwrap();
        assert_eq!(legacy, bilateral);
    }

    #[test]
    fn negative_bilateral_drive_reverses_side_phase_and_adhesion() {
        let gait = GaitLibrary::open("assets/neuromechfly/tripod_gait.json").unwrap();
        let reversed = gait.sample_bilateral(0.7, [-1.0, 1.0]).unwrap();
        let mirrored = gait.sample_bilateral(-0.7, [1.0, 1.0]).unwrap();
        assert_eq!(
            &reversed.joint_controls[..21],
            &mirrored.joint_controls[..21]
        );
        assert_eq!(&reversed.adhesion[..3], &mirrored.adhesion[..3]);
        assert_ne!(
            &reversed.joint_controls[21..],
            &mirrored.joint_controls[21..]
        );
    }
}
