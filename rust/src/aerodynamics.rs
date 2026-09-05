use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::TAU;
use std::fs;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

const AERODYNAMICS_FILE: &str = "aerodynamics.json";
const MODEL_SCHEMA: &str = "flybrain-aerodynamics-v1";
const TRANSLATIONAL_MODEL_NAME: &str = "translational_quasi_steady";
const MUJOCO_ELLIPSOID_MODEL_NAME: &str = "mujoco_ellipsoid";
const MUJOCO_ELLIPSOID_BACKEND: &str = "mujoco_ellipsoid";
const FLUIDCOEF_COUNT: usize = 5;
const FLYBODY_WINGBEAT_TYPE: &str = "flybody_fourier";
const FLYBODY_WINGBEAT_HARMONICS: usize = 12;
const FLYBODY_WINGBEAT_FREQUENCY_HZ: f64 = 218.0;
const VECTOR_TOLERANCE: f64 = 1e-6;

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct Vec3(pub [f64; 3]);

impl Vec3 {
    pub const ZERO: Self = Self([0.0, 0.0, 0.0]);

    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self([x, y, z])
    }

    pub fn x(self) -> f64 {
        self.0[0]
    }

    pub fn y(self) -> f64 {
        self.0[1]
    }

    pub fn z(self) -> f64 {
        self.0[2]
    }

    pub fn dot(self, other: Self) -> f64 {
        self.0
            .iter()
            .zip(other.0)
            .map(|(left, right)| left * right)
            .sum()
    }

    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y() * other.z() - self.z() * other.y(),
            self.z() * other.x() - self.x() * other.z(),
            self.x() * other.y() - self.y() * other.x(),
        )
    }

    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    pub fn normalized(self) -> Option<Self> {
        let norm = self.norm();
        if norm > VECTOR_TOLERANCE {
            Some(self / norm)
        } else {
            None
        }
    }

    pub fn is_finite(self) -> bool {
        self.0.iter().all(|value| value.is_finite())
    }

    pub fn reflected_y(self) -> Self {
        Self::new(self.x(), -self.y(), self.z())
    }
}

impl From<[f64; 3]> for Vec3 {
    fn from(value: [f64; 3]) -> Self {
        Self(value)
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|axis| self.0[axis] + rhs.0[axis]))
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|axis| self.0[axis] - rhs.0[axis]))
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0.map(|value| value * rhs))
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self(self.0.map(|value| value / rhs))
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct AerodynamicsConfig {
    pub schema: String,
    pub units: Units,
    pub model: ModelMetadata,
    pub air: AirProperties,
    pub coefficients: CoefficientModel,
    pub wingbeat: WingbeatConfig,
    pub wings: Vec<WingConfig>,
    pub limitations: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Units {
    pub length: String,
    pub time: String,
    pub mass: String,
    pub velocity: String,
    pub density: String,
    pub dynamic_viscosity: String,
    pub force: String,
    pub moment: String,
    pub angle: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    #[serde(default = "default_translational_backend")]
    pub backend: String,
    pub version: u32,
    pub strips_per_wing: usize,
    pub wings: Vec<String>,
    pub force_frame: String,
    pub input_frame: String,
    pub relative_velocity: String,
    pub dynamic_pressure: String,
    pub strip_force: String,
    pub strip_moment: String,
    pub air_velocity_mm_s: Vec3,
    pub force_application: String,
    #[serde(default)]
    pub fluid_geom_names: Vec<String>,
    #[serde(default)]
    pub fluidcoef: Option<[f64; FLUIDCOEF_COUNT]>,
}

fn default_translational_backend() -> String {
    TRANSLATIONAL_MODEL_NAME.to_owned()
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct AirProperties {
    pub rho_g_per_mm3: f64,
    pub dynamic_viscosity_g_per_mm_s: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CoefficientModel {
    pub source_fit: String,
    pub lift: CoefficientFit,
    pub drag: CoefficientFit,
    pub moment: MomentFit,
    pub angle_domain_rad: [f64; 2],
    pub outside_domain: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct CoefficientFit {
    pub offset: f64,
    pub amplitude: f64,
    pub angle_gain: f64,
    pub phase_rad: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MomentFit {
    pub formula: String,
    pub value: f64,
    pub status: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WingbeatConfig {
    pub frequency_hz: f64,
    pub waveform: WingbeatWaveform,
    pub joint_order: Vec<String>,
    pub phase_rad_by_side: BTreeMap<String, f64>,
    pub center_rad_by_axis: BTreeMap<String, f64>,
    pub amplitude_rad_by_axis: BTreeMap<String, f64>,
    pub mirror_roll_for_right: bool,
    pub joint_ranges_rad: BTreeMap<String, [f64; 2]>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WingbeatWaveform {
    #[serde(rename = "type", default = "default_waveform_type")]
    pub waveform_type: String,
    #[serde(default)]
    pub stroke_axis: String,
    #[serde(default)]
    pub stroke_center_rad: f64,
    #[serde(default)]
    pub stroke_amplitude_rad: f64,
    #[serde(default)]
    pub pitch_axis: String,
    #[serde(default)]
    pub pitch_center_rad: f64,
    #[serde(default)]
    pub pitch_amplitude_rad: f64,
    #[serde(default)]
    pub pitch_half_cycle_sign: bool,
    #[serde(default)]
    pub pitch_switch_basis: String,
    #[serde(default)]
    pub roll_axis: String,
    #[serde(default)]
    pub roll_center_rad: f64,
    #[serde(default)]
    pub roll_amplitude_rad: f64,
    #[serde(default)]
    pub fourier: Option<FourierWaveform>,
}

fn default_waveform_type() -> String {
    "legacy".to_owned()
}

#[derive(Clone, Debug, Deserialize)]
pub struct FourierWaveform {
    pub harmonics: usize,
    pub sample_count: usize,
    pub source_columns: Vec<String>,
    pub project_axes: Vec<String>,
    pub coefficient_convention: String,
    pub offset_rad_by_axis: BTreeMap<String, f64>,
    pub cos_rad_by_axis: BTreeMap<String, Vec<f64>>,
    pub sin_rad_by_axis: BTreeMap<String, Vec<f64>>,
    pub provenance: FourierProvenance,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FourierProvenance {
    pub source_file: String,
    pub source_sha256: String,
    pub doi: String,
    pub flybody_commit: String,
    pub fit_error: FourierFitError,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FourierFitError {
    pub max_abs_error_rad: f64,
    pub rmse_rad: f64,
    pub per_axis: BTreeMap<String, FourierAxisFitError>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FourierAxisFitError {
    pub max_abs_error_rad: f64,
    pub rmse_rad: f64,
    pub source_min_rad: f64,
    pub source_max_rad: f64,
    pub fit_min_rad: f64,
    pub fit_max_rad: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WingConfig {
    pub side: String,
    pub body: String,
    #[serde(default)]
    pub body_frame: Option<BodyFrame>,
    pub actuator_names: Vec<String>,
    pub strips: Vec<WingStrip>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BodyFrame {
    pub body_name: String,
    pub pos_mm: Vec3,
    pub quat_wxyz: [f64; 4],
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct WingStrip {
    pub index: usize,
    pub center_local_mm: Vec3,
    #[serde(default)]
    pub centroid_local_mm: Option<Vec3>,
    pub span_hat_local: Vec3,
    pub chord_hat_local: Vec3,
    pub normal_hat_local: Vec3,
    pub chord_mm: f64,
    pub width_mm: f64,
    #[serde(default)]
    pub area_mm2: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StripForce {
    pub relative_velocity_mm_s: Vec3,
    pub speed_mm_s: f64,
    pub angle_of_attack_rad: f64,
    pub cl: f64,
    pub cd: f64,
    pub dynamic_pressure_g_per_mm_s2: f64,
    pub area_mm2: f64,
    pub lift_direction: Vec3,
    pub drag_direction: Vec3,
    pub force_g_mm_s2: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WingTelemetry {
    pub side: String,
    pub force_g_mm_s2: Vec3,
    pub moment_g_mm2_s2: Vec3,
    pub strips: Vec<StripForce>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WingbeatCommand {
    pub time_s: f64,
    pub left: [f64; 3],
    pub right: [f64; 3],
}

#[derive(Clone, Debug)]
pub struct WingbeatGenerator {
    config: WingbeatConfig,
}

impl AerodynamicsConfig {
    pub fn load(path_or_dir: impl AsRef<Path>) -> Result<Self> {
        let path = path_or_dir.as_ref();
        let path = if path.is_dir() {
            path.join(AERODYNAMICS_FILE)
        } else {
            path.to_owned()
        };
        let bytes = fs::read(&path)
            .with_context(|| format!("reading aerodynamics specification {}", path.display()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing aerodynamics specification {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn open(path_or_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load(path_or_dir)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != MODEL_SCHEMA {
            bail!("unsupported aerodynamics schema: {}", self.schema)
        }
        validate_units(&self.units)?;
        validate_model(&self.model)?;
        validate_air(&self.air)?;
        validate_coefficients(&self.coefficients)?;
        self.wingbeat.validate()?;
        if self.limitations.is_empty() {
            bail!("aerodynamics limitations must be documented")
        }
        if self.wings.len() != 2 {
            bail!("aerodynamics requires exactly one left and one right wing")
        }
        let mut sides = BTreeSet::new();
        for wing in &self.wings {
            wing.validate(self.model.strips_per_wing)?;
            if !sides.insert(wing.side.as_str()) {
                bail!("duplicate wing side {}", wing.side)
            }
        }
        if sides != BTreeSet::from(["left", "right"]) {
            bail!("aerodynamics wings must contain left and right sides")
        }
        if self
            .model
            .wings
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != sides
        {
            bail!("model wing sides do not match wing entries")
        }
        Ok(())
    }

    pub fn strip_force(
        &self,
        strip: &WingStrip,
        relative_velocity_mm_s: Vec3,
    ) -> Result<StripForce> {
        if !relative_velocity_mm_s.is_finite() {
            bail!("relative strip velocity must be finite")
        }
        strip.validate(strip.index)?;
        let speed_mm_s = relative_velocity_mm_s.norm();
        let area_mm2 = strip.chord_mm * strip.width_mm;
        if speed_mm_s <= VECTOR_TOLERANCE {
            return Ok(StripForce {
                relative_velocity_mm_s,
                speed_mm_s: 0.0,
                angle_of_attack_rad: 0.0,
                cl: 0.0,
                cd: 0.0,
                dynamic_pressure_g_per_mm_s2: 0.0,
                area_mm2,
                lift_direction: Vec3::ZERO,
                drag_direction: Vec3::ZERO,
                force_g_mm_s2: Vec3::ZERO,
            });
        }
        let velocity_hat = relative_velocity_mm_s / speed_mm_s;
        let plane_velocity = relative_velocity_mm_s
            - strip.span_hat_local * relative_velocity_mm_s.dot(strip.span_hat_local);
        let angle_of_attack_rad = plane_velocity
            .dot(strip.normal_hat_local)
            .atan2(plane_velocity.dot(strip.chord_hat_local).abs());
        let cl = coefficient_value(self.coefficients.lift, angle_of_attack_rad, false);
        let cd = coefficient_value(self.coefficients.drag, angle_of_attack_rad, true);
        let lift_direction = (strip.normal_hat_local
            - velocity_hat * strip.normal_hat_local.dot(velocity_hat))
        .normalized()
        .unwrap_or(Vec3::ZERO);
        let drag_direction = -velocity_hat;
        let dynamic_pressure_g_per_mm_s2 = 0.5 * self.air.rho_g_per_mm3 * speed_mm_s * speed_mm_s;
        let force_g_mm_s2 =
            (drag_direction * cd + lift_direction * cl) * (dynamic_pressure_g_per_mm_s2 * area_mm2);
        Ok(StripForce {
            relative_velocity_mm_s,
            speed_mm_s,
            angle_of_attack_rad,
            cl,
            cd,
            dynamic_pressure_g_per_mm_s2,
            area_mm2,
            lift_direction,
            drag_direction,
            force_g_mm_s2,
        })
    }

    pub fn aggregate_wing(
        &self,
        wing_index: usize,
        relative_velocities_mm_s: &[Vec3],
    ) -> Result<WingTelemetry> {
        let wing = self
            .wings
            .get(wing_index)
            .with_context(|| format!("wing index {wing_index} is out of range"))?;
        self.aggregate_wing_for(wing, relative_velocities_mm_s)
    }

    pub fn aggregate_wing_for(
        &self,
        wing: &WingConfig,
        relative_velocities_mm_s: &[Vec3],
    ) -> Result<WingTelemetry> {
        if relative_velocities_mm_s.len() != wing.strips.len() {
            bail!(
                "wing {} has {} strips but {} velocities were supplied",
                wing.side,
                wing.strips.len(),
                relative_velocities_mm_s.len()
            )
        }
        let mut force_g_mm_s2 = Vec3::ZERO;
        let mut moment_g_mm2_s2 = Vec3::ZERO;
        let mut strips = Vec::with_capacity(wing.strips.len());
        for (strip, velocity) in wing.strips.iter().zip(relative_velocities_mm_s) {
            let strip_force = self.strip_force(strip, *velocity)?;
            force_g_mm_s2 = force_g_mm_s2 + strip_force.force_g_mm_s2;
            moment_g_mm2_s2 =
                moment_g_mm2_s2 + strip.center_local_mm.cross(strip_force.force_g_mm_s2);
            strips.push(strip_force);
        }
        Ok(WingTelemetry {
            side: wing.side.clone(),
            force_g_mm_s2,
            moment_g_mm2_s2,
            strips,
        })
    }

    pub fn wingbeat_generator(&self) -> Result<WingbeatGenerator> {
        WingbeatGenerator::new(self.wingbeat.clone())
    }

    pub fn uses_mujoco_ellipsoid(&self) -> bool {
        self.model.backend == MUJOCO_ELLIPSOID_BACKEND
    }
}

impl WingConfig {
    pub fn validate(&self, expected_strip_count: usize) -> Result<()> {
        if self.side != "left" && self.side != "right" {
            bail!("unsupported wing side {}", self.side)
        }
        if self.body.is_empty() {
            bail!("wing body name is empty")
        }
        if self.actuator_names.len() != 3
            || self.actuator_names.iter().any(String::is_empty)
            || self.actuator_names.iter().collect::<BTreeSet<_>>().len() != 3
        {
            bail!(
                "wing {} actuator_names must contain three unique names",
                self.side
            )
        }
        if self.strips.len() != expected_strip_count {
            bail!(
                "wing {} has {} strips; expected {}",
                self.side,
                self.strips.len(),
                expected_strip_count
            )
        }
        for (index, strip) in self.strips.iter().enumerate() {
            strip.validate(index)?;
        }
        if let Some(body_frame) = &self.body_frame {
            if body_frame.body_name != self.body
                || !body_frame.pos_mm.is_finite()
                || body_frame.quat_wxyz.iter().any(|value| !value.is_finite())
                || body_frame
                    .quat_wxyz
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    <= VECTOR_TOLERANCE
            {
                bail!("wing {} body frame is invalid", self.side)
            }
        }
        Ok(())
    }
}

impl WingStrip {
    pub fn validate(&self, expected_index: usize) -> Result<()> {
        if self.index != expected_index {
            bail!("wing strips must be indexed contiguously from zero")
        }
        if !self.center_local_mm.is_finite()
            || !self.span_hat_local.is_finite()
            || !self.chord_hat_local.is_finite()
            || !self.normal_hat_local.is_finite()
            || !self.chord_mm.is_finite()
            || !self.width_mm.is_finite()
            || self.chord_mm <= 0.0
            || self.width_mm <= 0.0
        {
            bail!("wing strip {} has invalid geometry", self.index)
        }
        for (name, vector) in [
            ("span_hat_local", self.span_hat_local),
            ("chord_hat_local", self.chord_hat_local),
            ("normal_hat_local", self.normal_hat_local),
        ] {
            if (vector.norm() - 1.0).abs() > VECTOR_TOLERANCE {
                bail!("wing strip {} {} is not unit length", self.index, name)
            }
        }
        if self.span_hat_local.dot(self.chord_hat_local).abs() > VECTOR_TOLERANCE
            || self.span_hat_local.dot(self.normal_hat_local).abs() > VECTOR_TOLERANCE
            || self.chord_hat_local.dot(self.normal_hat_local).abs() > VECTOR_TOLERANCE
            || self
                .span_hat_local
                .cross(self.chord_hat_local)
                .dot(self.normal_hat_local)
                .abs()
                < 1.0 - VECTOR_TOLERANCE
        {
            bail!("wing strip {} basis is not orthonormal", self.index)
        }
        if let Some(area_mm2) = self.area_mm2
            && (!area_mm2.is_finite() || area_mm2 <= 0.0)
        {
            bail!("wing strip {} area is invalid", self.index)
        }
        if let Some(centroid_local_mm) = self.centroid_local_mm
            && !centroid_local_mm.is_finite()
        {
            bail!("wing strip {} centroid is invalid", self.index)
        }
        Ok(())
    }
}

impl WingbeatConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            bail!("wingbeat frequency must be finite and positive")
        }
        if self
            .joint_order
            .iter()
            .map(String::as_str)
            .ne(["yaw", "pitch", "roll"])
        {
            bail!("wingbeat joint order must be yaw,pitch,roll")
        }
        if let Some(fourier) = &self.waveform.fourier {
            validate_fourier_waveform(self.frequency_hz, &self.waveform.waveform_type, fourier)?;
        } else {
            if self.waveform.stroke_axis != "yaw"
                || self.waveform.pitch_axis != "pitch"
                || self.waveform.roll_axis != "roll"
                || self.waveform.pitch_switch_basis != "stroke_velocity"
            {
                bail!("wingbeat waveform axes must be yaw,pitch,roll")
            }
            for (name, value) in [
                ("stroke_center_rad", self.waveform.stroke_center_rad),
                ("stroke_amplitude_rad", self.waveform.stroke_amplitude_rad),
                ("pitch_center_rad", self.waveform.pitch_center_rad),
                ("pitch_amplitude_rad", self.waveform.pitch_amplitude_rad),
                ("roll_center_rad", self.waveform.roll_center_rad),
                ("roll_amplitude_rad", self.waveform.roll_amplitude_rad),
            ] {
                if !value.is_finite() || (name.contains("amplitude") && value < 0.0) {
                    bail!("wingbeat {name} is invalid")
                }
            }
        }
        for side in ["left", "right"] {
            let phase = self
                .phase_rad_by_side
                .get(side)
                .with_context(|| format!("wingbeat phase is missing {side}"))?;
            if !phase.is_finite() {
                bail!("wingbeat phase for {side} is invalid")
            }
        }
        for axis in ["yaw", "pitch", "roll"] {
            let center = self
                .center_rad_by_axis
                .get(axis)
                .with_context(|| format!("wingbeat center is missing {axis}"))?;
            let amplitude = self
                .amplitude_rad_by_axis
                .get(axis)
                .with_context(|| format!("wingbeat amplitude is missing {axis}"))?;
            let range = self
                .joint_ranges_rad
                .get(axis)
                .with_context(|| format!("wingbeat joint range is missing {axis}"))?;
            if !center.is_finite()
                || !amplitude.is_finite()
                || *amplitude < 0.0
                || range.iter().any(|value| !value.is_finite())
                || range[0] >= range[1]
                || *center - *amplitude < range[0]
                || *center + *amplitude > range[1]
            {
                bail!("wingbeat {axis} center, amplitude, or range is invalid")
            }
        }
        Ok(())
    }
}

fn validate_fourier_waveform(
    frequency_hz: f64,
    waveform_type: &str,
    fourier: &FourierWaveform,
) -> Result<()> {
    if waveform_type != FLYBODY_WINGBEAT_TYPE
        || (frequency_hz - FLYBODY_WINGBEAT_FREQUENCY_HZ).abs() > 1e-12
        || fourier.harmonics != FLYBODY_WINGBEAT_HARMONICS
        || fourier.sample_count == 0
        || fourier.source_columns != ["yaw", "roll", "pitch"]
        || fourier.project_axes != ["yaw", "pitch", "roll"]
        || fourier.coefficient_convention.is_empty()
    {
        bail!("FlyBody Fourier wingbeat metadata is invalid")
    }
    for axis in ["yaw", "pitch", "roll"] {
        let offset = *fourier
            .offset_rad_by_axis
            .get(axis)
            .with_context(|| format!("Fourier offset is missing {axis}"))?;
        let cosine = fourier
            .cos_rad_by_axis
            .get(axis)
            .with_context(|| format!("Fourier cosine coefficients are missing {axis}"))?;
        let sine = fourier
            .sin_rad_by_axis
            .get(axis)
            .with_context(|| format!("Fourier sine coefficients are missing {axis}"))?;
        if !offset.is_finite()
            || cosine.len() != fourier.harmonics
            || sine.len() != fourier.harmonics
            || cosine.iter().chain(sine).any(|value| !value.is_finite())
        {
            bail!("Fourier coefficients are invalid for {axis}")
        }
        let fit = fourier
            .provenance
            .fit_error
            .per_axis
            .get(axis)
            .with_context(|| format!("Fourier fit error is missing {axis}"))?;
        if !fit.max_abs_error_rad.is_finite()
            || fit.max_abs_error_rad < 0.0
            || !fit.rmse_rad.is_finite()
            || fit.rmse_rad < 0.0
            || !fit.source_min_rad.is_finite()
            || !fit.source_max_rad.is_finite()
            || !fit.fit_min_rad.is_finite()
            || !fit.fit_max_rad.is_finite()
            || fit.source_min_rad > fit.source_max_rad
            || fit.fit_min_rad > fit.fit_max_rad
        {
            bail!("Fourier fit error is invalid for {axis}")
        }
    }
    let fit_error = &fourier.provenance.fit_error;
    if !fit_error.max_abs_error_rad.is_finite()
        || fit_error.max_abs_error_rad < 0.0
        || !fit_error.rmse_rad.is_finite()
        || fit_error.rmse_rad < 0.0
        || fourier.provenance.source_file.is_empty()
        || fourier.provenance.source_sha256.len() != 64
        || !fourier
            .provenance
            .source_sha256
            .chars()
            .all(|value| value.is_ascii_hexdigit())
        || fourier.provenance.doi.is_empty()
        || fourier.provenance.flybody_commit.is_empty()
    {
        bail!("Fourier wingbeat provenance is invalid")
    }
    Ok(())
}

impl WingbeatGenerator {
    pub fn new(config: WingbeatConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn command(&self, time_s: f64) -> Result<WingbeatCommand> {
        if !time_s.is_finite() {
            bail!("wingbeat time must be finite")
        }
        let left_phase =
            TAU * self.config.frequency_hz * time_s + self.config.phase_rad_by_side["left"];
        let right_phase =
            TAU * self.config.frequency_hz * time_s + self.config.phase_rad_by_side["right"];
        Ok(WingbeatCommand {
            time_s,
            left: self.side_command(left_phase, false),
            right: self.side_command(right_phase, true),
        })
    }

    pub fn config(&self) -> &WingbeatConfig {
        &self.config
    }

    fn side_command(&self, phase: f64, right: bool) -> [f64; 3] {
        if let Some(fourier) = &self.config.waveform.fourier {
            return [
                fourier_value(fourier, "yaw", phase),
                fourier_value(fourier, "pitch", phase),
                fourier_value(fourier, "roll", phase),
            ];
        }
        let stroke = self.config.waveform.stroke_center_rad
            + self.config.waveform.stroke_amplitude_rad * phase.sin();
        let pitch_sign = if self.config.waveform.pitch_half_cycle_sign {
            if phase.cos() >= 0.0 { 1.0 } else { -1.0 }
        } else {
            phase.cos()
        };
        let pitch = self.config.waveform.pitch_center_rad
            + self.config.waveform.pitch_amplitude_rad * pitch_sign;
        let roll_wave = self.config.waveform.roll_amplitude_rad * phase.sin();
        let roll = self.config.waveform.roll_center_rad
            + if right && self.config.mirror_roll_for_right {
                -roll_wave
            } else {
                roll_wave
            };
        [stroke, pitch, roll]
    }
}

fn fourier_value(waveform: &FourierWaveform, axis: &str, phase: f64) -> f64 {
    let mut value = waveform.offset_rad_by_axis[axis];
    for harmonic in 1..=waveform.harmonics {
        value += waveform.cos_rad_by_axis[axis][harmonic - 1] * (harmonic as f64 * phase).cos()
            + waveform.sin_rad_by_axis[axis][harmonic - 1] * (harmonic as f64 * phase).sin();
    }
    value
}

fn validate_units(units: &Units) -> Result<()> {
    let expected = [
        ("length", units.length.as_str(), "mm"),
        ("time", units.time.as_str(), "s"),
        ("mass", units.mass.as_str(), "g"),
        ("velocity", units.velocity.as_str(), "mm/s"),
        ("density", units.density.as_str(), "g/mm^3"),
        (
            "dynamic_viscosity",
            units.dynamic_viscosity.as_str(),
            "g/(mm*s)",
        ),
        ("force", units.force.as_str(), "g*mm/s^2"),
        ("moment", units.moment.as_str(), "g*mm^2/s^2"),
        ("angle", units.angle.as_str(), "rad"),
    ];
    for (name, actual, expected) in expected {
        if actual != expected {
            bail!("unsupported aerodynamics {name} unit: {actual}")
        }
    }
    Ok(())
}

fn validate_model(model: &ModelMetadata) -> Result<()> {
    if model.version != 1 || model.strips_per_wing == 0 {
        bail!("unsupported translational quasi-steady model metadata")
    }
    if model.backend == TRANSLATIONAL_MODEL_NAME {
        if model.name != TRANSLATIONAL_MODEL_NAME {
            bail!("translational backend has an incompatible model name")
        }
    } else if model.backend == MUJOCO_ELLIPSOID_BACKEND {
        if model.name != MUJOCO_ELLIPSOID_MODEL_NAME
            || model.fluid_geom_names.len() != 2
            || model.fluid_geom_names.iter().any(String::is_empty)
        {
            bail!("MuJoCo ellipsoid backend metadata is incomplete")
        }
        let fluidcoef = model
            .fluidcoef
            .ok_or_else(|| anyhow::anyhow!("MuJoCo ellipsoid backend is missing fluidcoef"))?;
        if fluidcoef.iter().any(|value| !value.is_finite()) || fluidcoef[0] <= 0.0 {
            bail!("MuJoCo ellipsoid fluidcoef is invalid")
        }
    } else {
        bail!("unsupported aerodynamics backend {}", model.backend)
    }
    if !model.air_velocity_mm_s.is_finite()
        || model.force_frame != "world"
        || model.input_frame != "wing_body_local_then_world"
        || model.relative_velocity != "wing_strip_velocity_minus_air_velocity"
    {
        bail!("invalid aerodynamics model frame metadata")
    }
    Ok(())
}

fn validate_air(air: &AirProperties) -> Result<()> {
    if !air.rho_g_per_mm3.is_finite()
        || air.rho_g_per_mm3 <= 0.0
        || !air.dynamic_viscosity_g_per_mm_s.is_finite()
        || air.dynamic_viscosity_g_per_mm_s < 0.0
    {
        bail!("air properties are invalid")
    }
    Ok(())
}

fn validate_coefficients(coefficients: &CoefficientModel) -> Result<()> {
    if coefficients.source_fit.is_empty()
        || !coefficients.lift.offset.is_finite()
        || !coefficients.lift.amplitude.is_finite()
        || !coefficients.lift.angle_gain.is_finite()
        || !coefficients.lift.phase_rad.is_finite()
        || !coefficients.drag.offset.is_finite()
        || !coefficients.drag.amplitude.is_finite()
        || !coefficients.drag.angle_gain.is_finite()
        || !coefficients.drag.phase_rad.is_finite()
        || !coefficients.moment.value.is_finite()
        || coefficients.moment.formula.is_empty()
        || coefficients.moment.status.is_empty()
        || coefficients.outside_domain.is_empty()
        || !coefficients.angle_domain_rad[0].is_finite()
        || !coefficients.angle_domain_rad[1].is_finite()
        || coefficients.angle_domain_rad[0] >= coefficients.angle_domain_rad[1]
    {
        bail!("coefficient model contains invalid values")
    }
    if coefficients.lift.angle_gain == 0.0 || coefficients.drag.angle_gain == 0.0 {
        bail!("coefficient angle gains must be non-zero")
    }
    Ok(())
}

fn coefficient_value(fit: CoefficientFit, alpha: f64, cosine: bool) -> f64 {
    let phase = fit.angle_gain * alpha + fit.phase_rad;
    if cosine {
        fit.offset + fit.amplitude * phase.cos()
    } else {
        fit.offset + fit.amplitude * phase.sin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AerodynamicsConfig {
        AerodynamicsConfig::load("assets/neuromechfly/aerodynamics.json").unwrap()
    }

    fn strip() -> WingStrip {
        WingStrip {
            index: 0,
            center_local_mm: Vec3::new(0.0, 1.0, 0.0),
            centroid_local_mm: None,
            span_hat_local: Vec3::new(0.0, 1.0, 0.0),
            chord_hat_local: Vec3::new(1.0, 0.0, 0.0),
            normal_hat_local: Vec3::new(0.0, 0.0, 1.0),
            chord_mm: 1.0,
            width_mm: 1.0,
            area_mm2: None,
        }
    }

    #[test]
    fn zero_speed_has_no_force() {
        let force = config().strip_force(&strip(), Vec3::ZERO).unwrap();
        assert_eq!(force.force_g_mm_s2, Vec3::ZERO);
        assert_eq!(force.dynamic_pressure_g_per_mm_s2, 0.0);
    }

    #[test]
    fn force_scales_with_speed_squared() {
        let config = config();
        let first = config
            .strip_force(&strip(), Vec3::new(10.0, 0.0, 10.0))
            .unwrap();
        let second = config
            .strip_force(&strip(), Vec3::new(20.0, 0.0, 20.0))
            .unwrap();
        assert!((second.force_g_mm_s2.norm() / first.force_g_mm_s2.norm() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn drag_reverses_with_velocity() {
        let config = config();
        let forward = config
            .strip_force(&strip(), Vec3::new(10.0, 0.0, 0.0))
            .unwrap();
        let backward = config
            .strip_force(&strip(), Vec3::new(-10.0, 0.0, 0.0))
            .unwrap();
        assert!(forward.force_g_mm_s2.x() < 0.0);
        assert!(backward.force_g_mm_s2.x() > 0.0);
        assert!((forward.force_g_mm_s2.x() + backward.force_g_mm_s2.x()).abs() < 1e-12);
    }

    #[test]
    fn mirrored_wings_have_mirrored_world_strip_frames() {
        let config = config();
        let left = config
            .wings
            .iter()
            .find(|wing| wing.side == "left")
            .unwrap();
        let right = config
            .wings
            .iter()
            .find(|wing| wing.side == "right")
            .unwrap();
        let left_strip = left.strips.first().unwrap();
        let right_strip = right.strips.first().unwrap();
        let left_frame = left.body_frame.as_ref().unwrap().quat_wxyz;
        let right_frame = right.body_frame.as_ref().unwrap().quat_wxyz;
        let left_span = quaternion_rotate(left_frame, left_strip.span_hat_local);
        let right_span = quaternion_rotate(right_frame, right_strip.span_hat_local);
        let left_chord = quaternion_rotate(left_frame, left_strip.chord_hat_local);
        let right_chord = quaternion_rotate(right_frame, right_strip.chord_hat_local);
        let left_normal = quaternion_rotate(left_frame, left_strip.normal_hat_local);
        let right_normal = quaternion_rotate(right_frame, right_strip.normal_hat_local);
        assert_vec3_close(left_span.reflected_y(), right_span);
        assert_vec3_close(left_chord.reflected_y(), right_chord);
        assert_vec3_close(-left_normal.reflected_y(), right_normal);
    }

    fn quaternion_rotate([w, x, y, z]: [f64; 4], vector: Vec3) -> Vec3 {
        let imaginary = Vec3::new(x, y, z);
        vector
            + imaginary.cross(vector) * (2.0 * w)
            + imaginary.cross(imaginary.cross(vector)) * 2.0
    }

    fn assert_vec3_close(left: Vec3, right: Vec3) {
        assert!(
            (left - right).norm() < 1e-12,
            "left={left:?}, right={right:?}"
        );
    }

    #[test]
    fn malformed_config_is_rejected() {
        let mut config = config();
        config.wings[0].strips[0].chord_hat_local = Vec3::new(1.0, 1.0, 0.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn wingbeat_is_finite_bounded_and_symmetric() {
        let config = config();
        let generator = config.wingbeat_generator().unwrap();
        for sample in 0..1000 {
            let time_s = sample as f64 / 997.0 / config.wingbeat.frequency_hz;
            let command = generator.command(time_s).unwrap();
            assert!(
                command
                    .left
                    .iter()
                    .chain(command.right.iter())
                    .all(|value| value.is_finite())
            );
            for axis in 0..3 {
                let range = config.wingbeat.joint_ranges_rad[&config.wingbeat.joint_order[axis]];
                assert!(command.left[axis] >= range[0] - 1e-12);
                assert!(command.left[axis] <= range[1] + 1e-12);
                assert!(command.right[axis] >= range[0] - 1e-12);
                assert!(command.right[axis] <= range[1] + 1e-12);
            }
            assert!((command.left[0] - command.right[0]).abs() < 1e-12);
            assert!((command.left[1] - command.right[1]).abs() < 1e-12);
            assert!((command.left[2] - command.right[2]).abs() < 1e-12);
        }
    }

    #[test]
    fn flybody_fourier_waveform_has_pinned_provenance() {
        let config = config();
        let waveform = config.wingbeat.waveform.fourier.as_ref().unwrap();
        assert_eq!(config.wingbeat.frequency_hz, 218.0);
        assert_eq!(waveform.harmonics, 12);
        assert_eq!(waveform.sample_count, 500);
        assert_eq!(
            waveform.source_columns,
            ["yaw", "roll", "pitch"]
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            waveform.project_axes,
            ["yaw", "pitch", "roll"]
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            waveform.provenance.source_sha256,
            "f97b975ef1b5adbe42c208ea9665f3c37fa432914f56f3943f1697cdb090d1ce"
        );
        assert_eq!(waveform.provenance.doi, "10.1038/s41586-025-09029-4");
        assert_eq!(
            waveform.provenance.flybody_commit,
            "d015e9bfe441bd90ae431bac24c55cb74bdbce26"
        );
        assert!(waveform.provenance.fit_error.max_abs_error_rad < 0.018);
    }

    #[test]
    fn flybody_fourier_waveform_is_periodic_and_has_expected_extrema() {
        let config = config();
        let generator = config.wingbeat_generator().unwrap();
        let period_s = 1.0 / config.wingbeat.frequency_hz;
        let first = generator.command(0.137 * period_s).unwrap();
        let second = generator.command(1.137 * period_s).unwrap();
        for axis in 0..3 {
            assert!((first.left[axis] - second.left[axis]).abs() < 1e-10);
            assert!((first.right[axis] - second.right[axis]).abs() < 1e-10);
            assert!((first.left[axis] - first.right[axis]).abs() < 1e-12);
        }

        let mut extrema = [[f64::INFINITY, f64::NEG_INFINITY]; 3];
        for sample in 0..4096 {
            let command = generator
                .command(sample as f64 * period_s / 4096.0)
                .unwrap();
            for (axis, extrema_axis) in extrema.iter_mut().enumerate() {
                extrema_axis[0] = extrema_axis[0].min(command.left[axis]);
                extrema_axis[1] = extrema_axis[1].max(command.left[axis]);
            }
        }
        for (axis, name) in ["yaw", "pitch", "roll"].into_iter().enumerate() {
            let center = config.wingbeat.center_rad_by_axis[name];
            let amplitude = config.wingbeat.amplitude_rad_by_axis[name];
            assert!(extrema[axis][0] < center);
            assert!(extrema[axis][1] > center);
            assert!(extrema[axis][0] >= center - amplitude - 2e-4);
            assert!(extrema[axis][1] <= center + amplitude + 2e-4);
        }
    }

    #[test]
    fn wing_aggregation_returns_force_and_moment() {
        let config = config();
        let velocities = vec![Vec3::new(10.0, 0.0, 0.0); config.wings[0].strips.len()];
        let telemetry = config.aggregate_wing(0, &velocities).unwrap();
        assert_eq!(telemetry.strips.len(), config.wings[0].strips.len());
        assert!(telemetry.force_g_mm_s2.is_finite());
        assert!(telemetry.moment_g_mm2_s2.is_finite());
    }

    #[test]
    fn vector_operations_are_consistent() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(x.dot(y), 0.0);
    }
}
